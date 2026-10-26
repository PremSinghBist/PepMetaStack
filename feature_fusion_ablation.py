#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import torch
from torch import nn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from metrics import Metric
import data_util


METRIC_NAMES = ("Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2")
FUSION_METHODS = ("concat", "attention", "cross_attention", "gated")


class TorchFusionEncoder(nn.Module):
    """Maps heterogeneous protein feature blocks to a common trainable latent space.

    One modality token is generated per pooled source vector. This operates on
    modality-level features, NOT on residue-level sequence tokens.
    """

    def __init__(self, input_dims, strategy, hidden_dim=64, num_heads=4):
        super().__init__()
        if strategy not in FUSION_METHODS[1:]:
            raise ValueError(f"Unsupported learned fusion strategy: {strategy}")
        if strategy == "cross_attention" and len(input_dims) < 2:
            raise ValueError("Cross attention needs ProtT5/first embedding and >=1 auxiliary modality.")
        if hidden_dim < 1 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads.")
        self.strategy = strategy
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            for dim in input_dims
        ])
        if strategy == "attention":
            self.attention_scores = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2 or 1),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2 or 1, 1),
            )
        elif strategy == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )
        elif strategy == "gated":
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def encode(self, blocks):
        tokens = torch.stack(
            [projection(x) for projection, x in zip(self.projections, blocks)], dim=1
        )  # [batch, modality, hidden]
        if self.strategy == "attention":
            weights = torch.softmax(self.attention_scores(tokens), dim=1)
            fused = (tokens * weights).sum(dim=1)
        elif self.strategy == "gated":
            gates = self.gate(tokens)  # independently gate each modality AND hidden dimension
            fused = (tokens * gates).sum(dim=1) / gates.sum(dim=1).clamp_min(1e-6)
        else:  # cross_attention: first embedding queries all other feature blocks
            query = tokens[:, :1, :]
            keys_values = tokens[:, 1:, :]
            attended, _ = self.cross_attention(
                query=query, key=keys_values, value=keys_values, need_weights=False
            )
            fused = (query + attended).squeeze(1)
        return self.output_norm(fused)

    def forward(self, blocks):
        return self.regression_head(self.encode(blocks)).squeeze(-1)


class LearnedFeatureFusion:
    """Supervised fusion trained ONLY on the fold's training rows/targets."""

    def __init__(self, method, config=None, random_state=42):
        config = dict(config or {})
        unknown = set(config) - {"hidden_dim", "num_heads", "epochs", "batch_size",
                                 "learning_rate", "weight_decay", "device"}
        if unknown:
            raise ValueError(f"Unknown fusion settings: {sorted(unknown)}")
        self.method = method
        self.hidden_dim = int(config.get("hidden_dim", 64))
        self.num_heads = int(config.get("num_heads", 4))
        self.epochs = int(config.get("epochs", 30))
        self.batch_size = int(config.get("batch_size", 64))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.device = torch.device(config.get("device", "cpu"))
        self.random_state = int(random_state)
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("Fusion epochs, batch_size, and learning_rate must be > 0.")

    def fit(self, feature_blocks, y):
        if not feature_blocks or len(y) == 0:
            raise ValueError("Fusion fit requires nonempty feature blocks and targets.")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.scalers_ = []
        arrays = []
        for block in feature_blocks:
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 2 or len(block) != len(y) or not np.isfinite(block).all():
                raise ValueError("Each feature block must be a finite 2D matrix with n rows.")
            scaler = StandardScaler().fit(block)
            self.scalers_.append(scaler)
            arrays.append(torch.from_numpy(scaler.transform(block).astype(np.float32)))
        if not np.isfinite(y).all():
            raise ValueError("Fusion training targets must be finite.")
        self.y_mean_ = float(np.mean(y))
        self.y_scale_ = max(float(np.std(y)), 1e-6)
        targets = torch.from_numpy((y - self.y_mean_) / self.y_scale_)

        # Seed every newly fitted fuser, including those used in inner cross fitting.
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        self.model_ = TorchFusionEncoder(
            [b.shape[1] for b in arrays], self.method,
            hidden_dim=self.hidden_dim, num_heads=self.num_heads,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        rng = np.random.default_rng(self.random_state)
        self.model_.train()
        for _ in range(self.epochs):
            for ids in np.array_split(
                rng.permutation(len(y)),
                max(1, int(np.ceil(len(y) / self.batch_size))),
            ):
                idx = torch.as_tensor(ids, dtype=torch.long)
                xb = [array.index_select(0, idx).to(self.device) for array in arrays]
                prediction = self.model_(xb)
                target_batch = targets.index_select(0, idx).to(self.device)
                loss = nn.functional.mse_loss(prediction, target_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

    def transform(self, feature_blocks):
        if len(feature_blocks) != len(self.scalers_):
            raise ValueError("Different number of feature modalities in fit and transform.")
        arrays = [
            torch.from_numpy(scaler.transform(np.asarray(block, dtype=np.float32)).astype(np.float32))
            for scaler, block in zip(self.scalers_, feature_blocks)
        ]
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("Feature modality row counts disagree.")
        output = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                batch = [array[start:start + self.batch_size].to(self.device) for array in arrays]
                output.append(self.model_.encode(batch).cpu().numpy())
        return np.concatenate(output, axis=0).astype(np.float32)


class CrossFittedFusionStack:
    """Fitted base estimators + cross-fitted RidgeCV meta learner + full-train fuser."""

    def __init__(self, fuser, estimators, fitted_estimators, final_estimator):
        self.fuser = fuser
        self.estimators = estimators  # naming convention expected by weight reporter
        self.fitted_estimators_ = fitted_estimators
        self.final_estimator_ = final_estimator

    def predict(self, blocks):
        features = self.fuser.transform(blocks)
        base_predictions = np.column_stack([
            estimator.predict(features) for estimator in self.fitted_estimators_
        ])
        return self.final_estimator_.predict(base_predictions)


class StackingRegressionPipeline:
    """Stacking regression with explicitly enabled sequence features."""

    def __init__(
        self,
        embedding_files=None,
        physio_path=None,
        blosum_path=None,
        sinu_path=None,
        random_state=42,
        *,
        pfeature_path=None,
        use_physio=False,
        use_blosum=False,
        use_sinusoidal=False,
        use_pfeature=False,
    ):
        """Flags alone enable extra features. Disabled paths are ignored.

        Pfeature files must be trusted .npy files containing a scalar dictionary
        mapping exact sequence strings to nonempty, finite 1D numeric vectors.
        Embedding selection remains controlled by embedding_sources at run time.
        """
        self.random_state = random_state
        self._dimensions = {}
        self.embeddings = {}
        self.feature_flags = dict(
            physio=use_physio,
            blosum=use_blosum,
            sinusoidal=use_sinusoidal,
            pfeature=use_pfeature,
        )
        self.feature_paths = dict(
            physio=physio_path,
            blosum=blosum_path,
            sinusoidal=sinu_path,
            pfeature=pfeature_path,
        )
        for name, flag in self.feature_flags.items():
            if type(flag) is not bool:
                raise TypeError(f"use_{name} must be True or False.")
        self.use_physio = use_physio
        self.use_blosum = use_blosum
        self.use_sinusoidal = use_sinusoidal
        self.use_pfeature = use_pfeature
        for name, path in (embedding_files or {}).items():
            self._require_file(path, f"Embedding {name}")
            self.embeddings[name] = self._validate_mapping(
                data_util.load_embeddings(path), f"embedding:{name}", pooled=True
            )
        self.extra_features = {}
        for name, enabled in self.feature_flags.items():
            if not enabled:
                continue
            path = self.feature_paths[name]
            self._require_file(path, name)
            if name == "physio":
                import physiochem_feature_extractor

                mapping = (
                    physiochem_feature_extractor.load_physio_features_as_numpy_all(path)
                )
            elif name == "blosum":
                import blosum62

                mapping = blosum62.load_blosum62_features(path)
            elif name == "sinusoidal":
                import position_encoding_extractor

                mapping = position_encoding_extractor.load_sinusoidal_encoding(path)
            else:
                if Path(path).suffix.lower() != ".npy":
                    raise ValueError("pfeature_path must reference a .npy file.")
                stored = np.load(path, allow_pickle=True)
                if not isinstance(stored, np.ndarray) or stored.shape != ():
                    raise ValueError("Pfeature .npy must contain a scalar dictionary.")
                mapping = stored.item()
            self.extra_features[name] = self._validate_mapping(
                mapping, name, strict_vector=name == "pfeature"
            )
        self.physio_features = self.extra_features.get("physio")
        self.blosum_features = self.extra_features.get("blosum")
        self.sinu_features = self.extra_features.get("sinusoidal")
        self.pfeature_features = self.extra_features.get("pfeature")
        self._print_configuration()

    @staticmethod
    def _require_file(path, name):
        if path is None or not str(path).strip():
            raise ValueError(f"{name} is enabled but its path is missing.")
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} file not found: {path}")

    def _validate_mapping(self, mapping, name, pooled=False, strict_vector=False):
        from collections.abc import Mapping

        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(
                f"{name}: expected a nonempty sequence-to-feature mapping."
            )
        dimension = None
        validated = {}
        for sequence, value in mapping.items():
            if not isinstance(sequence, str) or not sequence.strip():
                raise ValueError(f"{name}: keys must be nonempty sequence strings.")
            try:
                arr = np.asarray(value, dtype=np.float32)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(
                    f"{name}: nonnumeric features for {sequence!r}."
                ) from exc
            if arr.size == 0 or not np.isfinite(arr).all():
                raise ValueError(
                    f"{name}: empty or nonfinite features for {sequence!r}."
                )
            if strict_vector and arr.ndim != 1:
                raise ValueError(
                    f"{name}: expected 1D vector for {sequence!r}, got {arr.shape}."
                )
            vector = self.mean_pool(arr) if pooled else arr.ravel()
            if dimension is None:
                dimension = vector.size
            if vector.size != dimension:
                raise ValueError(
                    f"{name}: inconsistent dimensions for {sequence!r}: {vector.size}, expected {dimension}."
                )
            validated[sequence] = vector
        self._dimensions[name] = dimension
        return validated

    def _print_configuration(self):
        print("\n" + "=" * 70)
        print("STACKING REGRESSION FEATURE CONFIGURATION")
        for name, mapping in self.embeddings.items():
            print(
                f"Embedding {name}: {len(mapping)} sequences; {self._dimensions['embedding:' + name]} pooled dimensions"
            )
        print("Embeddings used are selected by embedding_sources for each run.")
        for name, enabled in self.feature_flags.items():
            detail = (
                f"path={self.feature_paths[name]}; dimensions={self._dimensions[name]}"
                if enabled
                else "path ignored"
            )
            print(f"{name}: {('ON' if enabled else 'OFF')}; {detail}")
        print(
            "Order: selected embeddings, physicochemical, BLOSUM62, sinusoidal, Pfeature."
        )
        print("Outer CV=5; inner stacking CV=5; fusion ablation=validation only.")
        print("=" * 70)

    def get_extra_features(self, sequence):
        features = []
        for name, mapping in self.extra_features.items():
            if sequence not in mapping:
                raise KeyError(f"{name}: sequence missing: {sequence!r}")
            features.append(mapping[sequence])
        return features

    def prepare_split(self, dataset_split, embedding_sources, pooling="mean"):
        if not isinstance(dataset_split, pd.DataFrame):
            raise TypeError("dataset_split must be a pandas DataFrame.")
        if not {"sequence", "value"}.issubset(dataset_split.columns):
            raise ValueError("Dataset requires sequence and value columns.")
        if dataset_split.empty:
            raise ValueError("Dataset split is empty.")
        if dataset_split["sequence"].isna().any():
            raise ValueError("Dataset contains missing sequences.")
        sequences = dataset_split["sequence"].tolist()
        if any((not isinstance(s, str) or not s.strip() for s in sequences)):
            raise ValueError(
                "Sequences must be nonempty strings matching feature keys exactly."
            )
        if isinstance(embedding_sources, str):
            raise TypeError("embedding_sources must be a list of source names.")
        if not embedding_sources or len(set(embedding_sources)) != len(
            embedding_sources
        ):
            raise ValueError("embedding_sources must be nonempty with no duplicates.")
        for source in embedding_sources:
            if source not in self.embeddings:
                raise ValueError(f"Unknown embedding source: {source}")
        selected = [(f"embedding:{s}", self.embeddings[s]) for s in embedding_sources]
        selected.extend(self.extra_features.items())
        for name, mapping in selected:
            missing = sorted(set(sequences) - mapping.keys())
            if missing:
                raise KeyError(
                    f"{name}: missing {len(missing)} sequences; examples: {missing[:5]}"
                )
        y = pd.to_numeric(dataset_split["value"], errors="raise").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(y).all():
            raise ValueError("Targets must be finite numeric values.")
        X = np.stack(
            [self.get_features(s, embedding_sources, pooling) for s in sequences]
        )
        print(f"Selected embeddings: {embedding_sources}; feature matrix: {X.shape}")
        return (X, y)

    @staticmethod
    def mean_pool(arr):
        """

        Mean-pool residue-level embeddings.

        Parameters

        ----------

        arr : np.ndarray

            Either:

                (sequence_length, embedding_dim)

            or:

                (embedding_dim,)

        Returns

        -------

        np.ndarray

            Shape:

                (embedding_dim,)

        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return np.mean(arr, axis=0).astype(np.float32)
        raise ValueError(
            f"Expected embedding to be 1D or 2D, but received shape {arr.shape}"
        )

    def get_embedding(self, sequence, source, pooling="mean"):
        """

        Retrieve and pool an embedding for one sequence.

        Parameters

        ----------

        sequence : str

            Protein sequence.

        source : str

            Embedding source name.

        pooling : str

            Currently only "mean".

        Returns

        -------

        np.ndarray

            Pooled embedding.

        """
        if source not in self.embeddings:
            raise ValueError(
                f"Embedding source '{source}' is not loaded.\nAvailable sources: {list(self.embeddings.keys())}"
            )
        embedding_dict = self.embeddings[source]
        if sequence not in embedding_dict:
            raise KeyError(
                f"Sequence not found in '{source}' embedding file:\n{sequence}"
            )
        embedding = embedding_dict[sequence]
        if pooling != "mean":
            raise ValueError(
                f"Unsupported pooling method: {pooling}. Currently only 'mean' is supported."
            )
        return self.mean_pool(embedding)

    def get_features(self, sequence, embedding_sources, pooling="mean"):
        """

        Build the complete feature vector for one sequence.

        Features consist of:

            Selected embeddings

                    +

            Optional physicochemical features

                    +

            Optional BLOSUM62 features

                    +

            Optional sinusoidal features

        All features are concatenated.

        Example

        -------

        embedding_sources=["prott5"]

            ProtT5 + optional handcrafted features

        embedding_sources=["prott5", "esm2"]

            ProtT5 + ESM2 + optional handcrafted features

        embedding_sources=["fusion"]

            Fusion embedding + optional handcrafted features

        """
        if not embedding_sources:
            raise ValueError("embedding_sources cannot be empty.")
        features = []
        for source in embedding_sources:
            feature = self.get_embedding(
                sequence=sequence, source=source, pooling=pooling
            )
            features.append(feature)
        features.extend(self.get_extra_features(sequence))
        if not features:
            raise ValueError("No features were generated.")
        return np.concatenate(features, axis=0).astype(np.float32)

    def build_stacking_ensemble(self):
        """

        Build stacking regression ensemble.

        Base estimators:

            1. Ridge

            2. SVR

            3. Extra Trees

            4. HistGradientBoosting

            5. KNN

            6. MLP

            7. XGBoost

        Meta learner:

            RidgeCV

        Scaling is performed inside the estimators that

        require it.

        """
        base_estimators = [
            # (
            #     "ridge",
            #     Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
            # ),
            (
                "svr",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
                    ]
                ),
            ),
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=16,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "hist_gb",
                HistGradientBoostingRegressor(
                    max_iter=150, l2_regularization=1.0, random_state=self.random_state
                ),
            ),
            (
                "knn",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            KNeighborsRegressor(
                                n_neighbors=7,
                                weights="distance",
                                metric="cosine",
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "mlp",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(128, 64),
                                activation="relu",
                                alpha=0.01,
                                max_iter=300,
                                early_stopping=True,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                ),
            ),
            # (
            #     "xgb",
            #     XGBRegressor(
            #         n_estimators=150,
            #         max_depth=5,
            #         learning_rate=0.05,
            #         subsample=0.8,
            #         colsample_bytree=0.8,
            #         random_state=self.random_state,
            #         n_jobs=-1,
            #     ),
            # ),
        ]
        meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 15))
        internal_cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=internal_cv,
            n_jobs=-1,
        )

    def prepare_split_blocks(self, dataset_split, embedding_sources, pooling="mean"):
        """Extract per-modality matrices for learned fusion (same vectors as concat)."""
        if not isinstance(dataset_split, pd.DataFrame) or dataset_split.empty:
            raise ValueError("Dataset split must be a nonempty DataFrame.")
        if not {"sequence", "value"}.issubset(dataset_split.columns):
            raise ValueError("Dataset split must have 'sequence' and 'value' columns.")
        if not embedding_sources or len(embedding_sources) != len(set(embedding_sources)):
            raise ValueError("Choose one or more distinct embedding sources.")
        if isinstance(embedding_sources, str):
            raise TypeError("embedding_sources must be a list of names, not a string.")
        for source in embedding_sources:
            if source not in self.embeddings:
                raise ValueError(f"Unknown embedding: {source}")
        seqs = dataset_split["sequence"].tolist()
        if any(not isinstance(s, str) or not s.strip() for s in seqs):
            raise ValueError("Missing or invalid protein sequence.")
        y = pd.to_numeric(dataset_split["value"], errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Targets must be finite.")
        names_and_mappings = (
            [(f"embedding:{source}", self.embeddings[source]) for source in embedding_sources]
            + list(self.extra_features.items())
        )
        blocks = []
        for name, mapping in names_and_mappings:
            missing = [seq for seq in seqs if seq not in mapping]
            if missing:
                raise KeyError(f"{name}: {len(missing)} missing feature vectors, e.g. {missing[:3]}")
            blocks.append(np.stack([mapping[s] for s in seqs]).astype(np.float32))
        return blocks, y

    def fit_cross_fitted_fusion_stack(self, train_blocks, y_train, method,
                                     fusion_config=None, fold_number=1):
        """Learn supervised fusion INSIDE each inner CV split.

        This matters: learning the fusion representation once on all outer-train
        labels before StackingRegressor's inner CV would leak inner holdout labels
        into meta-learner OOF predictions. Here each inner fold refits its own
        fuser, while a final fuser fits all outer-train data for validation inference.
        """
        if method not in FUSION_METHODS[1:]:
            raise ValueError(f"Expected learned fusion strategy, got {method}")
        if method == "cross_attention" and len(train_blocks) < 2:
            raise ValueError("Cross attention needs first embedding plus an auxiliary modality.")
        template = self.build_stacking_ensemble()
        estimators = template.estimators
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        n = len(y_train)
        if n < 5:
            raise ValueError("At least 5 outer-training samples required for inner CV.")
        oof = np.full((n, len(estimators)), np.nan, dtype=np.float64)
        salt = {"attention": 1, "cross_attention": 2, "gated": 3}[method]
        for inner_fold, (train_idx, holdout_idx) in enumerate(cv.split(np.arange(n)), start=1):
            seed = self.random_state + 10000 * fold_number + 100 * inner_fold + salt
            fuser = LearnedFeatureFusion(method, fusion_config, random_state=seed)
            fuser.fit([block[train_idx] for block in train_blocks], y_train[train_idx])
            train_features = fuser.transform([block[train_idx] for block in train_blocks])
            holdout_features = fuser.transform([block[holdout_idx] for block in train_blocks])
            for col, (name, estimator) in enumerate(estimators):
                fitted = clone(estimator).fit(train_features, y_train[train_idx])
                oof[holdout_idx, col] = fitted.predict(holdout_features)
            print(f"    inner fold {inner_fold}/5 fitted ({method})")
        if not np.isfinite(oof).all():
            raise RuntimeError("Incomplete inner out-of-fold base predictions.")
        meta = clone(template.final_estimator).fit(oof, y_train)
        full_fuser = LearnedFeatureFusion(
            method, fusion_config,
            random_state=self.random_state + 10000 * fold_number + 999 + salt,
        ).fit(train_blocks, y_train)
        full_features = full_fuser.transform(train_blocks)
        fitted_estimators = [
            clone(estimator).fit(full_features, y_train)
            for name, estimator in estimators
        ]
        return CrossFittedFusionStack(full_fuser, estimators, fitted_estimators, meta)

    def run_fusion_ablation(self, dataset_csv, metric_evaluator, embedding_sources,
                            fusion_methods=FUSION_METHODS, fusion_config=None,
                            pooling="mean"):
        """Evaluate each fusion method on the SAME five outer validation folds.

        Independent test records are inspected for split consistency only; no
        model fits, selects hyperparameters, or evaluates on test observations.
        """
        if not Path(dataset_csv).is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
        if pooling != "mean":
            raise ValueError("Only mean pooling is supported.")
        fusion_methods = tuple(fusion_methods)
        if not fusion_methods or len(fusion_methods) != len(set(fusion_methods)):
            raise ValueError("fusion_methods must be nonempty without duplicates.")
        if set(fusion_methods) - set(FUSION_METHODS):
            raise ValueError(f"Unknown fusion strategies: {set(fusion_methods) - set(FUSION_METHODS)}")

        folds = []
        reference_test_sequences = None
        reference_dev_sequences = None
        for fold_number in range(1, 6):
            train_df, val_df, test_df = data_util.get_cv_split(
                input_csv_path=dataset_csv, fold=fold_number
            )
            tr_seq = set(train_df["sequence"])
            val_seq = set(val_df["sequence"])
            test_seq = set(test_df["sequence"])
            if tr_seq & val_seq or (tr_seq | val_seq) & test_seq:
                raise ValueError(f"Sequence overlap detected in fold {fold_number}.")
            if reference_test_sequences is None:
                reference_test_sequences = test_seq
                reference_dev_sequences = tr_seq | val_seq
            elif test_seq != reference_test_sequences or (tr_seq | val_seq) != reference_dev_sequences:
                raise ValueError("Development/independent-test sets changed between folds.")
            train_blocks, y_train = self.prepare_split_blocks(
                train_df, embedding_sources, pooling
            )
            val_blocks, y_val = self.prepare_split_blocks(
                val_df, embedding_sources, pooling
            )
            if any(a.shape[1] != b.shape[1] for a, b in zip(train_blocks, val_blocks)):
                raise ValueError("Train and validation dimensions disagree.")
            folds.append((train_blocks, y_train, val_blocks, y_val))

        method_results = {}
        for method in fusion_methods:
            print("\n" + "=" * 72)
            print(f"Fusion ablation: {method} | random seed: {self.random_state}")
            print("=" * 72)
            metrics_by_fold = []
            for fold_number, (train_blocks, y_train, val_blocks, y_val) in enumerate(folds, 1):
                print(f"  outer fold {fold_number}/5: train={len(y_train)}, val={len(y_val)}")
                if method == "concat":
                    train_x = np.concatenate(train_blocks, axis=1)
                    val_x = np.concatenate(val_blocks, axis=1)
                    model = self.build_stacking_ensemble().fit(train_x, y_train)
                    predictions = model.predict(val_x)
                else:
                    model = self.fit_cross_fitted_fusion_stack(
                        train_blocks, y_train, method, fusion_config, fold_number
                    )
                    predictions = model.predict(val_blocks)
                scores = metric_evaluator.compute_metrics(
                    np.asarray(y_val, dtype=np.float64),
                    np.asarray(predictions, dtype=np.float64).reshape(-1)
                )
                metrics_by_fold.append(scores)
                print(f"    validation: " + ", ".join(
                    f"{metric}={float(scores[metric]):.4f}"
                    for metric in METRIC_NAMES if metric in scores
                ))
            summary = self.summarize_metrics(metrics_by_fold)
            print(f"\n{method} five-fold mean +/- sample SD:\n{summary.to_string(index=False)}")
            method_results[method] = {
                "validation_metrics": metrics_by_fold,
                "validation_summary": summary,
            }
        return {"random_seed": self.random_state, "methods": method_results}

    @staticmethod
    def save_fusion_ablation_reports(seed_results, output_dir,
                                     prefix="fusion_ablation"):
        """Persist per-fold scores, five-fold scores per seed, and cross-seed report."""
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(seed_results, dict):
            seed_results = [seed_results]
        fold_rows = []
        per_seed_rows = []
        for result in seed_results:
            seed = result["random_seed"]
            for method, method_result in result["methods"].items():
                for fold_idx, scores in enumerate(method_result["validation_metrics"], 1):
                    fold_rows.append({"Seed": seed, "Fusion": method,
                                      "Fold": fold_idx, **scores})
                row = {"Seed": seed, "Fusion": method, "n_folds": len(method_result["validation_metrics"])}
                for _, score in method_result["validation_summary"].iterrows():
                    key = score["Metric"]
                    avg, std = float(score["Mean"]), float(score["Std"])
                    row[f"{key}_Mean"] = avg
                    row[f"{key}_Std"] = std
                    row[f"{key}_Mean_Std"] = f"{avg:.4f} +/- {std:.4f}"
                per_seed_rows.append(row)
        fold_df = pd.DataFrame(fold_rows)
        seed_df = pd.DataFrame(per_seed_rows)
        report_rows = []
        for method in dict.fromkeys(seed_df["Fusion"]):
            subset = seed_df[seed_df["Fusion"] == method]
            for metric in METRIC_NAMES:
                col = f"{metric}_Mean"
                if col not in subset:
                    continue
                values = subset[col].to_numpy(dtype=np.float64)
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                report_rows.append({
                    "Fusion": method, "Metric": metric,
                    "Mean": mean, "Std": std,
                    "Mean_Std": f"{mean:.4f} +/- {std:.4f}" if len(values) > 1
                    else f"{mean:.4f} +/- N/A (only one seed)",
                    "n_seeds": len(values), "outer_folds_per_seed": 5,
                    "variability_definition": "SD of per-seed five-fold CV means",
                })
        report_df = pd.DataFrame(report_rows)
        files = {
            "folds": os.path.join(output_dir, f"{prefix}_validation_fold_metrics.csv"),
            "per_seed": os.path.join(output_dir, f"{prefix}_validation_seed_summary.csv"),
            "overall": os.path.join(output_dir, f"{prefix}_validation_report.csv"),
        }
        fold_df.to_csv(files["folds"], index=False)
        seed_df.to_csv(files["per_seed"], index=False)
        report_df.to_csv(files["overall"], index=False)
        print("\nSaved fusion-ablation reports:")
        for file in files.values():
            print(f"  {file}")
        return files

    def run_stacking_pipeline(
        self, dataset_csv, metric_evaluator: Metric, embedding_sources, pooling="mean"
    ):
        """

        Evaluation protocol:

        1. Perform outer five-fold cross-validation on the

        development dataset only.

        2. Within each outer training fold, StackingRegressor

        performs internal five-fold cross-validation to generate

        out-of-fold predictions for training the meta-learner.

        3. The fixed independent test set is not evaluated during

        outer cross-validation.

        4. After outer CV is completed, train a new stacking model

        using the complete development dataset.

        5. Evaluate the final model exactly once on the independent

        test set.

        """
        if not os.path.exists(dataset_csv):
            raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
        if not embedding_sources:
            raise ValueError("embedding_sources cannot be empty.")
        missing_sources = [
            source for source in embedding_sources if source not in self.embeddings
        ]
        if missing_sources:
            raise ValueError(
                f"Requested embedding sources are not loaded: {missing_sources}\nAvailable: {list(self.embeddings.keys())}"
            )
        print("\n" + "=" * 70)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 70)
        print(f"Embedding sources: {embedding_sources}")
        print(f"Pooling: {pooling}")
        print("=" * 70)
        validation_metrics = []
        fold_results = []
        development_dataset = None
        independent_test = None
        development_sequences = None
        independent_sequences = None
        for fold_number in range(1, 6):
            print("\n" + "=" * 70)
            print(f"OUTER FOLD {fold_number}/5")
            print("=" * 70)
            train_set, val_set, test_set = data_util.get_cv_split(
                input_csv_path=dataset_csv, fold=fold_number
            )
            train_set = train_set.copy().reset_index(drop=True)
            val_set = val_set.copy().reset_index(drop=True)
            test_set = test_set.copy().reset_index(drop=True)
            print(f"Training   : {len(train_set)}")
            print(f"Validation : {len(val_set)}")
            if fold_number == 1:
                development_dataset = pd.concat([train_set, val_set], ignore_index=True)
                independent_test = test_set.copy()
                development_sequences = set(
                    development_dataset["sequence"].astype(str).tolist()
                )
                independent_sequences = set(
                    independent_test["sequence"].astype(str).tolist()
                )
                overlap = development_sequences & independent_sequences
                if overlap:
                    raise ValueError(
                        f"Data leakage detected: {len(overlap)} sequences occur in both development and independent-test datasets."
                    )
                print(
                    "\nNo exact sequence overlap detected between development and independent-test datasets."
                )
                print(f"Complete development dataset : {len(development_dataset)}")
                print(f"Independent test dataset     : {len(independent_test)}")
            current_test_sequences = set(test_set["sequence"].astype(str).tolist())
            if current_test_sequences != independent_sequences:
                raise ValueError(
                    f"Independent test set changed across outer folds. Mismatch detected at fold {fold_number}."
                )
            print("\nPreparing training features...")
            X_train, y_train = self.prepare_split(
                train_set, embedding_sources=embedding_sources, pooling=pooling
            )
            print("Preparing validation features...")
            X_val, y_val = self.prepare_split(
                val_set, embedding_sources=embedding_sources, pooling=pooling
            )
            if X_train.shape[1] != X_val.shape[1]:
                raise ValueError(
                    "Training and validation feature dimensions do not match."
                )
            print(f"\nFeature dimension: {X_train.shape[1]}")
            model = self.build_stacking_ensemble()
            print("\nTraining stacking ensemble...")
            model.fit(X_train, y_train)
            print("Training completed.")
            val_pred = model.predict(X_val)
            y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
            val_pred = np.asarray(val_pred, dtype=np.float64).reshape(-1)
            if len(y_val) != len(val_pred):
                raise ValueError("Validation target/prediction length mismatch.")
            val_metric = metric_evaluator.compute_metrics(y_val, val_pred)
            validation_metrics.append(val_metric)
            print(f"\nFold {fold_number} Validation Results:")
            self.print_metrics(val_metric)
            stacking_weights = self.analyze_stacking_weights(model, fold_number)
            fold_results.append(
                {
                    "fold": fold_number,
                    "val_y_true": y_val,
                    "val_y_pred": val_pred,
                    "stacking_weights": stacking_weights,
                }
            )
        print("\n" + "=" * 70)
        print("FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        validation_summary = self.summarize_metrics(validation_metrics)
        print("\nValidation performance (mean +/- SD):")
        print(validation_summary.to_string(index=False))
        print("\n" + "=" * 70)
        print("TRAINING FINAL MODEL ON COMPLETE DEVELOPMENT DATASET")
        print("=" * 70)
        X_full, y_full = self.prepare_split(
            development_dataset, embedding_sources=embedding_sources, pooling=pooling
        )
        print(f"\nComplete development feature matrix: {X_full.shape}")
        final_model = self.build_stacking_ensemble()
        final_model.fit(X_full, y_full)
        print("Final model training completed.")
        final_stacking_weights = self.analyze_stacking_weights(final_model, "FINAL")
        print("\n" + "=" * 70)
        print("FINAL INDEPENDENT TEST EVALUATION")
        print("=" * 70)
        X_test, y_test = self.prepare_split(
            independent_test, embedding_sources=embedding_sources, pooling=pooling
        )
        if X_full.shape[1] != X_test.shape[1]:
            raise ValueError(
                "Development and independent-test feature dimensions do not match."
            )
        test_pred = final_model.predict(X_test)
        y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
        test_pred = np.asarray(test_pred, dtype=np.float64).reshape(-1)
        if len(y_test) != len(test_pred):
            raise ValueError("Independent-test target/prediction length mismatch.")
        independent_metric = metric_evaluator.compute_metrics(y_test, test_pred)
        print("\nFINAL Independent Test Results:")
        self.print_metrics(independent_metric)
        return {
            "validation_metrics": validation_metrics,
            "validation_summary": validation_summary,
            "fold_results": fold_results,
            "final_model": final_model,
            "final_stacking_weights": final_stacking_weights,
            "independent_test_metric": independent_metric,
            "independent_y_true": y_test,
            "independent_y_pred": test_pred,
        }

    @staticmethod
    def print_metrics(metrics):
        """

        Print regression metrics in a consistent order.

        """
        ordered_keys = ["Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2"]
        for key in ordered_keys:
            if key in metrics:
                print(f"  {key:<15}: {metrics[key]:.4f}")

    @staticmethod
    def summarize_metrics(metrics_list):
        """

        Calculate mean and sample standard deviation

        across the five outer folds.

        """
        metric_names = ["Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2"]
        rows = []
        for metric_name in metric_names:
            values = [
                metric[metric_name] for metric in metrics_list if metric_name in metric
            ]
            if not values:
                continue
            rows.append(
                {
                    "Metric": metric_name,
                    "Mean": np.mean(values),
                    "Std": np.std(values, ddof=1),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def analyze_stacking_weights(stacking_model, fold_idx):
        """

        Extract meta-learner coefficients.

        Coefficients represent the relationship between

        base-model predictions and the final RidgeCV

        prediction.

        Absolute coefficient magnitude should NOT be

        interpreted as direct predictive importance or

        percentage contribution.

        """
        estimator_names = [name for name, _ in stacking_model.estimators]
        meta_learner = stacking_model.final_estimator_
        coefficients = np.asarray(meta_learner.coef_).ravel()
        if len(coefficients) != len(estimator_names):
            raise ValueError(
                "Number of meta-learner coefficients does not match number of base estimators."
            )
        weights_df = pd.DataFrame(
            {
                "Base_Model": estimator_names,
                "Coefficient": coefficients,
                "Absolute_Weight": np.abs(coefficients),
            }
        )
        total = weights_df["Absolute_Weight"].sum()
        if total > 0:
            weights_df["Coefficient_Abs_Percent"] = (
                weights_df["Absolute_Weight"] / total * 100
            )
        else:
            weights_df["Coefficient_Abs_Percent"] = 0.0
        weights_df = weights_df.sort_values(
            "Absolute_Weight", ascending=False
        ).reset_index(drop=True)
        print(f"\nMeta-learner weights — Fold {fold_idx}:")
        print(weights_df.to_string(index=False))
        intercept = float(np.asarray(meta_learner.intercept_).ravel()[0])
        print(f"\nMeta learner intercept: {intercept:.4f}")
        return weights_df

    @staticmethod
    def save_results(results, output_dir, prefix="stacking"):
        """

        Save:

        1. Outer five-fold validation summary

        2. Outer fold validation metrics

        3. Final independent-test metrics

        4. Final independent-test predictions

        5. Outer-fold meta-learner coefficients

        6. Final-model meta-learner coefficients

        """
        os.makedirs(output_dir, exist_ok=True)
        validation_summary_file = os.path.join(
            output_dir, f"{prefix}_validation_summary.csv"
        )
        results["validation_summary"].to_csv(validation_summary_file, index=False)
        validation_rows = []
        for fold_idx, metrics in enumerate(results["validation_metrics"], start=1):
            row = {"Fold": fold_idx}
            row.update(metrics)
            validation_rows.append(row)
        validation_df = pd.DataFrame(validation_rows)
        validation_file = os.path.join(
            output_dir, f"{prefix}_validation_fold_metrics.csv"
        )
        validation_df.to_csv(validation_file, index=False)
        independent_file = os.path.join(
            output_dir, f"{prefix}_final_independent_test_metrics.csv"
        )
        pd.DataFrame([results["independent_test_metric"]]).to_csv(
            independent_file, index=False
        )
        prediction_file = os.path.join(
            output_dir, f"{prefix}_final_independent_predictions.csv"
        )
        pd.DataFrame(
            {
                "y_true": results["independent_y_true"],
                "y_pred": results["independent_y_pred"],
            }
        ).to_csv(prediction_file, index=False)
        weight_rows = []
        for fold_result in results["fold_results"]:
            weights_df = fold_result["stacking_weights"].copy()
            weights_df.insert(0, "Fold", fold_result["fold"])
            weight_rows.append(weights_df)
        if weight_rows:
            all_weights_df = pd.concat(weight_rows, ignore_index=True)
            weights_file = os.path.join(output_dir, f"{prefix}_cv_meta_weights.csv")
            all_weights_df.to_csv(weights_file, index=False)
            print(f"Saved CV meta weights: {weights_file}")
        final_weights_file = os.path.join(
            output_dir, f"{prefix}_final_meta_weights.csv"
        )
        results["final_stacking_weights"].to_csv(final_weights_file, index=False)
        print(f"\nSaved validation summary: {validation_summary_file}")
        print(f"Saved validation fold metrics: {validation_file}")
        print(f"Saved final independent metrics: {independent_file}")
        print(f"Saved final independent predictions: {prediction_file}")
        print(f"Saved final meta weights: {final_weights_file}")


if __name__ == "__main__":
    DATASET_CSV = "./data/ecoli_five_fold_dataset.csv"
    EMBEDDING_FILES = {
        "prott5": "./data/prott5/prot_t5_xl_uniref50_ecoli.npz",
        "ft_prott5": "./data/prott5/ecoli_finetuned_prott5.npz",
        "esm2": "./data/esm2/esm2_t33_650M_UR50D_1280_ecoli.npz",
    }
    EMBEDDING_SOURCES = ["prott5"]

    # Feature selection is identical for all fusion strategies.
    USE_PHYSIO = True
    USE_BLOSUM = True
    USE_SINUSOIDAL = True
    USE_PFEATURE = False
    PFEATURE_PATH = "./data/ecoli_pfeatures.npy"
    PHYSIO_PATH = "./data/ecoli_phyiochem.csv"
    BLOSUM_PATH = "./data/ecoli_blosum62_features.csv"
    SINU_PATH = "./data/ecoli_sinusoidal_encoding.csv"

    EXPERIMENT_NAME = "PEPSTACK_ECOLI_FUSION_ABLATION"
    OUTPUT_ROOT = "./results/ecoli_fusion_ablation"
    RAND_SEED = [42] #[42, 12, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    METHODS_TO_COMPARE = ["concat", "attention", "cross_attention", "gated"]
    FUSION_CONFIG = {
        "hidden_dim": 64,       # all learned fusion outputs have 64 dimensions
        "num_heads": 4,         # multi-head cross-attention
        "epochs": 30,           # increase/tune using development data if desired
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "device": "cuda",        # cpu: change to "cuda" if a CUDA-enabled GPU is available
    }

    # Load large embedding and handcrafted feature dictionaries only once.
    pipeline = StackingRegressionPipeline(
            embedding_files={key: EMBEDDING_FILES[key] for key in EMBEDDING_SOURCES},
            physio_path=PHYSIO_PATH,
            blosum_path=BLOSUM_PATH,
            sinu_path=SINU_PATH,
            random_state=RAND_SEED[0],
            pfeature_path=PFEATURE_PATH,
            use_physio=USE_PHYSIO,
            use_blosum=USE_BLOSUM,
            use_sinusoidal=USE_SINUSOIDAL,
            use_pfeature=USE_PFEATURE,
    )
    metric = Metric("RegressionMetrics")
    all_seed_results = []
    for random_seed in RAND_SEED:
        pipeline.random_state = random_seed
        result = pipeline.run_fusion_ablation(
            dataset_csv=DATASET_CSV,
            metric_evaluator=metric,
            embedding_sources=EMBEDDING_SOURCES,
            fusion_methods=METHODS_TO_COMPARE,
            fusion_config=FUSION_CONFIG,
        )
        all_seed_results.append(result)

        # Save as each seed finishes so partial experiments remain available.
        seed_output = os.path.join(OUTPUT_ROOT, f"seed_{random_seed}")
        pipeline.save_fusion_ablation_reports(result, seed_output)
        pipeline.save_fusion_ablation_reports(all_seed_results, OUTPUT_ROOT)

    print("\nAll ablation experiments completed. Independent test set was not evaluated.")
