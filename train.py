#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
        print("Outer CV=5; inner stacking CV=5; final fit=full development; test=once.")
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

    # Explicit switches: paths alone never enable a feature.
    USE_PHYSIO = True
    USE_BLOSUM = True
    USE_SINUSOIDAL = True
    USE_PFEATURE = False
    PFEATURE_PATH = "./data/ecoli_pfeatures.npy"
    PHYSIO_PATH = "./data/ecoli_phyiochem.csv"
    BLOSUM_PATH = "./data/ecoli_blosum62_features.csv"
    SINU_PATH = "./data/ecoli_sinusoidal_encoding.csv"

    EXPERIMENT_NAME = "PEPSTACK_ECOLI_RAND_SEED"
    RAND_SEED = [42, 12, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    for i in range(len(RAND_SEED)):
        OUTPUT_PATH = f"./results/ecoli_random_seed/{EXPERIMENT_NAME}_{RAND_SEED[i]}"
        
        pipeline = StackingRegressionPipeline(
            embedding_files=EMBEDDING_FILES,
            physio_path=PHYSIO_PATH,
            blosum_path=BLOSUM_PATH,
            sinu_path=SINU_PATH,
            random_state=RAND_SEED[i],
            pfeature_path=PFEATURE_PATH,
            use_physio=USE_PHYSIO,
            use_blosum=USE_BLOSUM,
            use_sinusoidal=USE_SINUSOIDAL,
            use_pfeature=USE_PFEATURE,
        )
        metric = Metric("RegressionMetrics")
        results = pipeline.run_stacking_pipeline(
            dataset_csv=DATASET_CSV,
            metric_evaluator=metric,
            embedding_sources=EMBEDDING_SOURCES,
            pooling="mean",
        )
        pipeline.save_results(results, output_dir=OUTPUT_PATH, prefix="stacking")
