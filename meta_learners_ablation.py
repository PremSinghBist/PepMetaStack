#!/usr/bin/env python3
"""PepMetaStack meta-learner ablations (six alternatives).

Keeps all seven original base predictors unchanged and caches their five-fold
OOF predictions once per outer fold to compare second-stage learners fairly.
Run from the project directory containing metrics.py, data_util.py, and data/.
"""
from pathlib import Path

import os
import json
import time
from collections import Counter
from sklearn.base import clone

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, LassoCV, ElasticNetCV

from sklearn.svm import SVR

from sklearn.model_selection import KFold, cross_val_predict

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

class MetaLearnerAblationPipeline:

    """Stacking meta-learner ablations with explicitly enabled sequence features."""

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

        print("Outer CV=5; internal OOF CV=5; independent test optional (once).")

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

    def build_stacking_ensemble(self, selected_predictors=None, stack_n_jobs=1):

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

            (

                "ridge",

                Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),

            ),

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

            (

                "xgb",

                XGBRegressor(

                    n_estimators=150,

                    max_depth=5,

                    learning_rate=0.05,

                    subsample=0.8,

                    colsample_bytree=0.8,

                    random_state=self.random_state,

                    n_jobs=-1,

                ),

            ),

        ]

        if selected_predictors is not None:
            selected_predictors = tuple(selected_predictors)
            available = {name for name, _ in base_estimators}
            if (not selected_predictors or len(selected_predictors) != len(set(selected_predictors))
                    or any(name not in available for name in selected_predictors)):
                raise ValueError(f"Invalid base predictor selection: {selected_predictors}")
            selected_set = set(selected_predictors)
            base_estimators = [(name, estimator) for name, estimator in base_estimators
                               if name in selected_set]
        if len(base_estimators) < 2:
            raise ValueError("Stacking requires at least two base predictors; use the standalone estimator for A1.")

        meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 15))

        internal_cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        return StackingRegressor(

            estimators=base_estimators,

            final_estimator=meta_learner,

            cv=internal_cv,

            n_jobs=stack_n_jobs,

        )


    @staticmethod
    def metric_columns():
        return ["Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2"]

    def build_meta_learners(self):
        """Six competing second-level estimators; base learners never change.

        Pipeline scaling for SVR/Lasso/ElasticNet operates ONLY on base predictions,
        with scalers fitted on each corresponding outer training fold.
        """
        return {
            "ridge_cv": RidgeCV(alphas=np.logspace(-3, 3, 15)),
            "linear_regression": LinearRegression(),
            "lasso_cv": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LassoCV(
                    alphas=np.logspace(-4, 1, 60), cv=5, max_iter=10000,
                    random_state=self.random_state, n_jobs=1,
                )),
            ]),
            "elasticnet_cv": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    alphas=np.logspace(-4, 1, 60), cv=5, max_iter=10000,
                    random_state=self.random_state, n_jobs=1,
                )),
            ]),
            "svr_rbf": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(C=1.0, epsilon=0.1, kernel="rbf")),
            ]),
            "xgboost": XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=self.random_state,
                n_jobs=1,
            ),
        }

    def base_predictors(self):
        # Reuse the ORIGINAL seven estimators verbatim (including hyperparameters).
        # StackingRegressor is only used here as an estimator factory.
        return self.build_stacking_ensemble(stack_n_jobs=1).estimators

    @staticmethod
    def _save_csv(data, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        pd.DataFrame(data).to_csv(tmp, index=False)
        os.replace(tmp, target)

    @classmethod
    def _summarize(cls, fold_rows):
        metrics = cls.metric_columns()
        frame = pd.DataFrame(fold_rows)
        if len(frame) != 5:
            raise ValueError(f"Expected five validation folds, got {len(frame)}")
        summary = {"n_folds": len(frame)}
        for key in metrics:
            if key not in frame.columns:
                raise KeyError(f"Metric evaluator omitted {key}")
            values = frame[key].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {key} values: {values}")
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
            summary[f"{key}_mean_pm_std"] = f"{mean:.4f} ± {std:.4f}"
        return summary

    @staticmethod
    def _model_attributes(meta, base_names, fold):
        """Export available coefficients without mislabeling them as importance."""
        actual = meta.named_steps["model"] if isinstance(meta, Pipeline) else meta
        rows = []
        if hasattr(actual, "coef_"):
            coef = np.ravel(actual.coef_)
            if len(coef) == len(base_names):
                for name, value in zip(base_names, coef):
                    rows.append({
                        "Fold": fold, "Base_Model": name,
                        "Coefficient": float(value),
                        "Coefficient_Space": (
                            "standardized_meta_input" if isinstance(meta, Pipeline)
                            else "original_base_predictions"
                        ),
                    })
        attributes = {"Fold": fold, "Meta_Model": type(actual).__name__}
        for attr in ("alpha_", "l1_ratio_", "intercept_"):
            if hasattr(actual, attr):
                value = getattr(actual, attr)
                if np.size(value) == 1:
                    attributes[attr] = float(np.ravel(value)[0])
        return rows, attributes

    def _base_level_predictions(self, X_train, y_train, X_val, base_estimators,
                                inner_splits=5, stack_n_jobs=1):
        """Generate OOF training and full-fit validation meta-features.

        This reproduces the default StackingRegressor(passthrough=False) design:
        each meta-training column comes from cross_val_predict on the outer
        training fold, and each outer-validation column comes from its base
        regressor fitted on the FULL outer training fold.
        """
        if len(X_train) < inner_splits:
            raise ValueError("Too few training samples for internal CV")
        cv = KFold(n_splits=inner_splits, shuffle=True, random_state=self.random_state)
        oof_columns, val_columns = [], []
        for name, estimator in base_estimators:
            print(f"    Base predictor: {name}", flush=True)
            oof = cross_val_predict(
                clone(estimator), X_train, y_train, cv=cv,
                method="predict", n_jobs=stack_n_jobs,
            )
            full_base = clone(estimator).fit(X_train, y_train)
            val_predictions = full_base.predict(X_val)
            oof_columns.append(np.asarray(oof, dtype=np.float64).ravel())
            val_columns.append(np.asarray(val_predictions, dtype=np.float64).ravel())
        Z_train = np.column_stack(oof_columns)
        Z_val = np.column_stack(val_columns)
        if not np.isfinite(Z_train).all() or not np.isfinite(Z_val).all():
            raise ValueError("Nonfinite base-level predictions")
        return Z_train, Z_val

    def run_meta_learner_ablations(
        self, dataset_csv, metric_evaluator, embedding_sources,
        pooling="mean", output_root="results/meta_learners_ablation",
        stack_n_jobs=1, evaluate_independent=False, independent_model="ridge_cv",
    ):
        """SINGLE ENTRY POINT: six meta-learners / five-fold development CV.

        All six methods share exactly the same seven base regressors and OOF
        predictions within each fold. The independent test remains untouched
        by default and can optionally be run ONCE for a preselected model.
        """
        output_root = Path(output_root)
        if not Path(dataset_csv).is_file():
            raise FileNotFoundError(f"Missing dataset: {dataset_csv}")
        if not embedding_sources or any(s not in self.embeddings for s in embedding_sources):
            raise ValueError("Embedding sources must be loaded and nonempty")
        if not isinstance(stack_n_jobs, int) or stack_n_jobs == 0:
            raise ValueError("stack_n_jobs must be a nonzero integer")

        meta_factories = self.build_meta_learners()
        if independent_model not in meta_factories:
            raise ValueError(f"Unknown independent meta-learner: {independent_model}")
        base_estimators = self.base_predictors()
        base_names = [name for name, _ in base_estimators]
        all_fold_rows = []
        all_prediction_rows = []
        per_model_fold = {name: [] for name in meta_factories}
        coefficient_rows = {name: [] for name in meta_factories}
        parameter_rows = {name: [] for name in meta_factories}
        development_set = independent_test = None
        expected_development = expected_independent = None
        validation_counts = Counter()
        output_root.mkdir(parents=True, exist_ok=True)
        config = {
            "dataset_csv": str(Path(dataset_csv).resolve()),
            "embedding_sources": list(embedding_sources), "pooling": pooling,
            "feature_flags": self.feature_flags,
            "feature_paths": {key: str(path) if self.feature_flags[key] else None
                              for key, path in self.feature_paths.items()},
            "base_predictors": base_names,
            "meta_learners": list(meta_factories),
            "random_state": self.random_state,
            "outer_cv": 5, "internal_cv": 5,
            "meta_passthrough": False,
            "evaluate_independent": evaluate_independent,
            "independent_model": independent_model if evaluate_independent else None,
            "note": "Independent test is NOT used to compare or select meta-learners.",
        }
        (output_root / "run_configuration.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nMeta-learner ablation: {len(meta_factories)} methods × 5 outer folds")
        print(f"Output root: {output_root.resolve()}")

        def records(frame):
            # Multiset permits duplicate sequence/value rows.
            return Counter(zip(frame["sequence"].tolist(), frame["value"].tolist()))

        for fold in range(1, 6):
            print("\n" + "=" * 70)
            print(f"OUTER FOLD {fold}/5", flush=True)
            train, val, test = data_util.get_cv_split(
                input_csv_path=dataset_csv, fold=fold
            )
            train = train.copy().reset_index(drop=True)
            val = val.copy().reset_index(drop=True)
            test = test.copy().reset_index(drop=True)
            if fold == 1:
                development_set = pd.concat([train, val], ignore_index=True)
                independent_test = test
                expected_development = records(development_set)
                expected_independent = records(independent_test)
                overlap = set(development_set["sequence"]) & set(independent_test["sequence"])
                if overlap:
                    raise ValueError(f"Development/test sequence overlap: {len(overlap)}")
            else:
                if records(pd.concat([train, val], ignore_index=True)) != expected_development:
                    raise ValueError(f"Development population changed in fold {fold}")
                if records(test) != expected_independent:
                    raise ValueError(f"Independent test changed in fold {fold}")
            if set(train["sequence"]) & set(val["sequence"]):
                raise ValueError(f"Train/validation sequence overlap in fold {fold}")
            validation_counts.update(records(val))
            print(f"Train={len(train)}  Validation={len(val)}  Test held out={len(test)}")
            X_train, y_train = self.prepare_split(train, embedding_sources, pooling)
            X_val, y_val = self.prepare_split(val, embedding_sources, pooling)
            if X_train.shape[1] != X_val.shape[1]:
                raise ValueError("Inconsistent feature dimensions")
            Z_train, Z_val = self._base_level_predictions(
                X_train, y_train, X_val, base_estimators,
                inner_splits=5, stack_n_jobs=stack_n_jobs,
            )
            for name, meta_template in meta_factories.items():
                meta = clone(meta_template)
                meta.fit(Z_train, y_train)
                pred = np.asarray(meta.predict(Z_val), dtype=float).ravel()
                if len(pred) != len(y_val) or not np.isfinite(pred).all():
                    raise ValueError(f"Invalid predictions for {name} fold {fold}")
                metrics = metric_evaluator.compute_metrics(
                    np.asarray(y_val, dtype=np.float64), pred
                )
                row = {"Fold": fold, **{key: float(metrics[key])
                                       for key in self.metric_columns()}}
                per_model_fold[name].append(row)
                all_fold_rows.append({"Meta_Learner": name, **row})
                pred_frame = pd.DataFrame({
                    "Fold": fold, "sequence": val["sequence"].to_numpy(),
                    "y_true": np.asarray(y_val, dtype=float), "y_pred": pred,
                })
                pred_frame.insert(0, "Meta_Learner", name)
                all_prediction_rows.append(pred_frame)
                folder = output_root / name
                self._save_csv(per_model_fold[name], folder / "validation_fold_metrics.csv")
                self._save_csv(pred_frame, folder / f"fold_{fold}_predictions.csv")
                coef, params = self._model_attributes(meta, base_names, fold)
                coefficient_rows[name].extend(coef)
                parameter_rows[name].append(params)
                print(f"  {name:20s} RMSE={row['RMSE']:.4f}  R2={row['R2']:.4f}", flush=True)

        if validation_counts != expected_development:
            raise ValueError("Outer validation folds do not partition development data")

        summary_rows = []
        for name in meta_factories:
            folder = output_root / name
            summary = {"Meta_Learner": name, **self._summarize(per_model_fold[name])}
            summary_rows.append(summary)
            self._save_csv([summary], folder / "validation_summary_mean_std.csv")
            self._save_csv(parameter_rows[name], folder / "meta_model_selected_parameters.csv")
            if coefficient_rows[name]:
                self._save_csv(coefficient_rows[name], folder / "meta_coefficients.csv")
        summary_frame = pd.DataFrame(summary_rows).sort_values("RMSE_mean").reset_index(drop=True)
        self._save_csv(summary_frame, output_root / "all_meta_learners_mean_std.csv")
        self._save_csv(all_fold_rows, output_root / "all_meta_learners_fold_metrics.csv")
        self._save_csv(pd.concat(all_prediction_rows, ignore_index=True),
                       output_root / "all_meta_learners_cv_predictions.csv")
        print("\n" + "=" * 70)
        print("META-LEARNER CROSS-VALIDATION SUMMARY (mean ± sample SD)")
        display_cols = ["Meta_Learner"] + [f"{k}_mean_pm_std" for k in self.metric_columns()]
        print(summary_frame[display_cols].to_string(index=False))

        if evaluate_independent:
            # Explicit preselection only. Do not choose by independent-test metrics.
            print(f"\nFinal held-out evaluation for preselected {independent_model} only...")
            X_full, y_full = self.prepare_split(development_set, embedding_sources, pooling)
            X_test, y_test = self.prepare_split(independent_test, embedding_sources, pooling)
            Z_full, Z_test = self._base_level_predictions(
                X_full, y_full, X_test, base_estimators,
                inner_splits=5, stack_n_jobs=stack_n_jobs,
            )
            meta = clone(meta_factories[independent_model]).fit(Z_full, y_full)
            pred = np.asarray(meta.predict(Z_test), dtype=float).ravel()
            test_metric = metric_evaluator.compute_metrics(
                np.asarray(y_test, dtype=np.float64), pred
            )
            folder = output_root / independent_model
            self._save_csv([{"Meta_Learner": independent_model, **test_metric}],
                           folder / "final_independent_test_metrics.csv")
            self._save_csv({
                "sequence": independent_test["sequence"],
                "y_true": y_test, "y_pred": pred,
            }, folder / "final_independent_test_predictions.csv")
            print("Independent test (one preselected model):", test_metric)

        print(f"\nResults saved to: {output_root.resolve()}")
        return summary_frame


if __name__ == "__main__":
    # ========================================================================
    # EDIT CONFIGURATION HERE (matching your original PepMetaStack script)
    # ========================================================================
    DATASET_CSV = "./data/ecoli_five_fold_dataset.csv"
    EMBEDDING_FILES = {
        "prott5": "./data/prott5/prot_t5_xl_uniref50_ecoli.npz",
        "ft_prott5": "./data/prott5/ecoli_finetuned_prott5.npz",
        "esm2": "./data/esm2/esm2_t33_650M_UR50D_1280_ecoli.npz",
    }
    EMBEDDING_SOURCES = ["prott5"]
    USE_PHYSIO = True
    USE_BLOSUM = True
    USE_SINUSOIDAL = True
    USE_PFEATURE = False
    PFEATURE_PATH = "./data/ecoli_pfeatures.npy"
    PHYSIO_PATH = "./data/ecoli_phyiochem.csv"
    BLOSUM_PATH = "./data/ecoli_blosum62_features.csv"
    SINU_PATH = "./data/ecoli_sinusoidal_encoding.csv"
    OUTPUT_ROOT = "results/meta_learners_ablation"
    STACK_N_JOBS = 1
    # Disable during meta-learner selection: evaluate the independent test only
    # after the model has been preselected from development CV results.
    EVALUATE_INDEPENDENT = False
    INDEPENDENT_MODEL = "ridge_cv"
    # ========================================================================

    pipeline = MetaLearnerAblationPipeline(
        embedding_files=EMBEDDING_FILES,
        physio_path=PHYSIO_PATH,
        blosum_path=BLOSUM_PATH,
        sinu_path=SINU_PATH,
        random_state=42,
        pfeature_path=PFEATURE_PATH,
        use_physio=USE_PHYSIO,
        use_blosum=USE_BLOSUM,
        use_sinusoidal=USE_SINUSOIDAL,
        use_pfeature=USE_PFEATURE,
    )
    metric = Metric("RegressionMetrics")
    pipeline.run_meta_learner_ablations(
        dataset_csv=DATASET_CSV,
        metric_evaluator=metric,
        embedding_sources=EMBEDDING_SOURCES,
        pooling="mean",
        output_root=OUTPUT_ROOT,
        stack_n_jobs=STACK_N_JOBS,
        evaluate_independent=EVALUATE_INDEPENDENT,
        independent_model=INDEPENDENT_MODEL,
    )
