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
import physiochem_feature_extractor
import blosum62
import position_encoding_extractor


# ============================================================
# Stacking Regression Pipeline
# ============================================================

class StackingRegressionPipeline:

    """
    Generic stacking regression pipeline for protein sequence features.

    The pipeline supports:

        - ProtT5
        - Fine-tuned ProtT5
        - ESM2
        - Pre-computed fusion embeddings
        - Multiple embedding sources simultaneously
        - Physicochemical features
        - BLOSUM62 features
        - Sinusoidal positional features

    Examples
    --------
    Pure ProtT5:

        embedding_files = {
            "prott5": "prott5.npz"
        }

        embedding_sources = ["prott5"]


    Fine-tuned ProtT5:

        embedding_files = {
            "ft_prott5": "ft_prott5.npz"
        }

        embedding_sources = ["ft_prott5"]


    ProtT5 + ESM2:

        embedding_files = {
            "prott5": "prott5.npz",
            "esm2": "esm2.npz"
        }

        embedding_sources = [
            "prott5",
            "esm2"
        ]


    Pre-computed fusion embedding:

        embedding_files = {
            "fusion": "fusion.npz"
        }

        embedding_sources = ["fusion"]

    Handcrafted features are added automatically when their
    corresponding paths are supplied to the constructor.
    """

    # ========================================================
    # 1. INITIALIZATION
    # ========================================================

    def __init__(
        self,
        embedding_files=None,
        physio_path=None,
        blosum_path=None,
        sinu_path=None,
        random_state=42,
    ):

        self.random_state = random_state

        # ----------------------------------------------------
        # Load embedding files
        # ----------------------------------------------------

        self.embeddings = {}

        for name, path in (embedding_files or {}).items():

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Embedding file not found: {path}"
                )

            print(
                f"Loading embedding [{name}]: {path}"
            )

            self.embeddings[name] = (
                data_util.load_embeddings(path)
            )

        # ----------------------------------------------------
        # Load optional handcrafted features
        # ----------------------------------------------------

        self.physio_features = None
        self.blosum_features = None
        self.sinu_features = None

        if physio_path is not None:

            if not os.path.exists(physio_path):
                raise FileNotFoundError(
                    f"Physicochemical feature file not found: "
                    f"{physio_path}"
                )

            print(
                f"Loading physicochemical features: "
                f"{physio_path}"
            )

            self.physio_features = (
                physiochem_feature_extractor
                .load_physio_features_as_numpy_all(
                    physio_path
                )
            )

        if blosum_path is not None:

            if not os.path.exists(blosum_path):
                raise FileNotFoundError(
                    f"BLOSUM feature file not found: "
                    f"{blosum_path}"
                )

            print(
                f"Loading BLOSUM62 features: "
                f"{blosum_path}"
            )

            self.blosum_features = (
                blosum62.load_blosum62_features(
                    blosum_path
                )
            )

        if sinu_path is not None:

            if not os.path.exists(sinu_path):
                raise FileNotFoundError(
                    f"Sinusoidal feature file not found: "
                    f"{sinu_path}"
                )

            print(
                f"Loading sinusoidal features: "
                f"{sinu_path}"
            )

            self.sinu_features = (
                position_encoding_extractor
                .load_sinusoidal_encoding(
                    sinu_path
                )
            )

        self._print_configuration()

    # ========================================================
    # 2. CONFIGURATION
    # ========================================================

    def _print_configuration(self):

        print("\n" + "=" * 70)
        print("STACKING REGRESSION PIPELINE")
        print("=" * 70)

        print(
            "\nEmbedding sources loaded:"
        )

        if self.embeddings:

            for name, embedding in self.embeddings.items():

                print(
                    f"  - {name:<15} "
                    f"{len(embedding):>6} sequences"
                )

        else:

            print("  None")

        print("\nAdditional features:")

        print(
            f"  - Physicochemical : "
            f"{self.physio_features is not None}"
        )

        print(
            f"  - BLOSUM62        : "
            f"{self.blosum_features is not None}"
        )

        print(
            f"  - Sinusoidal      : "
            f"{self.sinu_features is not None}"
        )

        print("=" * 70)

    # ========================================================
    # 3. EMBEDDING POOLING
    # ========================================================

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

        arr = np.asarray(
            arr,
            dtype=np.float32,
        )

        # Already pooled
        if arr.ndim == 1:
            return arr

        # Residue-level embedding
        if arr.ndim == 2:

            return np.mean(
                arr,
                axis=0,
            ).astype(np.float32)

        raise ValueError(
            "Expected embedding to be 1D or 2D, "
            f"but received shape {arr.shape}"
        )

    # ========================================================
    # 4. GET EMBEDDING
    # ========================================================

    def get_embedding(
        self,
        sequence,
        source,
        pooling="mean",
    ):

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
                f"Embedding source '{source}' is not loaded.\n"
                f"Available sources: "
                f"{list(self.embeddings.keys())}"
            )
            
        print(f"Embedding source used for trainin is : {source}")
        embedding_dict = self.embeddings[source]

        if sequence not in embedding_dict:

            raise KeyError(
                f"Sequence not found in "
                f"'{source}' embedding file:\n"
                f"{sequence}"
            )

        embedding = embedding_dict[sequence]

        if pooling != "mean":

            raise ValueError(
                f"Unsupported pooling method: {pooling}. "
                "Currently only 'mean' is supported."
            )

        return self.mean_pool(embedding)

    # ========================================================
    # 5. GET ADDITIONAL FEATURES
    # ========================================================

    def get_extra_features(self, sequence):

        """
        Retrieve all optional handcrafted features.

        Returns
        -------
        list[np.ndarray]
            List of feature vectors.
        """

        features = []

        feature_sources = [
            self.physio_features,
            self.blosum_features,
            self.sinu_features,
        ]

        for feature_dict in feature_sources:

            if feature_dict is None:
                continue

            if sequence not in feature_dict:

                raise KeyError(
                    "Sequence not found in one of the "
                    "handcrafted feature files:\n"
                    f"{sequence}"
                )

            feature = np.asarray(
                feature_dict[sequence],
                dtype=np.float32,
            ).ravel()

            features.append(feature)

        return features

    # ========================================================
    # 6. BUILD FEATURES FOR ONE SEQUENCE
    # ========================================================

    def get_features(
        self,
        sequence,
        embedding_sources,
        pooling="mean",
    ):

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

            raise ValueError(
                "embedding_sources cannot be empty."
            )

        features = []

        # ----------------------------------------------------
        # Embedding features
        # ----------------------------------------------------

        for source in embedding_sources:

            feature = self.get_embedding(
                sequence=sequence,
                source=source,
                pooling=pooling,
            )

            features.append(feature)

        # ----------------------------------------------------
        # Handcrafted features
        # ----------------------------------------------------

        features.extend(
            self.get_extra_features(sequence)
        )

        if not features:

            raise ValueError(
                "No features were generated."
            )

        return np.concatenate(
            features,
            axis=0,
        ).astype(np.float32)

    # ========================================================
    # 7. PREPARE DATASET SPLIT
    # ========================================================

    def prepare_split(
        self,
        dataset_split,
        embedding_sources,
        pooling="mean",
    ):

        """
        Convert a DataFrame into X and y.

        Required columns:

            sequence
            value
        """
        
        print(f"The embeddings sources are: {embedding_sources}")

        if not isinstance(
            dataset_split,
            pd.DataFrame,
        ):

            raise TypeError(
                "dataset_split must be a pandas DataFrame."
            )

        required_columns = {
            "sequence",
            "value",
        }

        missing_columns = (
            required_columns
            - set(dataset_split.columns)
        )

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        X_list = []
        y_list = []

        for row in dataset_split.itertuples(
            index=False
        ):

            sequence = str(row.sequence)

            feature = self.get_features(
                sequence=sequence,
                embedding_sources=embedding_sources,
                pooling=pooling,
            )

            X_list.append(feature)
            y_list.append(
                float(row.value)
            )

        if not X_list:

            raise ValueError(
                "Dataset split is empty."
            )

        X = np.asarray(
            X_list,
            dtype=np.float32,
        )

        y = np.asarray(
            y_list,
            dtype=np.float32,
        )

        print(
            f"Feature matrix shape: {X.shape}"
        )

        return X, y

    # ========================================================
    # 8. BUILD STACKING ENSEMBLE
    # ========================================================

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

            # ------------------------------------------------
            # Ridge
            # ------------------------------------------------

            (
                "ridge",

                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),

                        (
                            "model",
                            Ridge(
                                alpha=10.0,
                            ),
                        ),
                    ]
                ),
            ),

            # ------------------------------------------------
            # SVR
            # ------------------------------------------------

            (
                "svr",

                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),

                        (
                            "model",
                            SVR(
                                C=10.0,
                                epsilon=0.1,
                                kernel="rbf",
                            ),
                        ),
                    ]
                ),
            ),

            # ------------------------------------------------
            # Extra Trees
            # ------------------------------------------------

            (
                "extra_trees",

                ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=16,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),

            # ------------------------------------------------
            # HistGradientBoosting
            # ------------------------------------------------

            (
                "hist_gb",

                HistGradientBoostingRegressor(
                    max_iter=150,
                    l2_regularization=1.0,
                    random_state=self.random_state,
                ),
            ),

            # ------------------------------------------------
            # KNN
            # ------------------------------------------------

            (
                "knn",

                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),

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

            # ------------------------------------------------
            # MLP
            # ------------------------------------------------

            (
                "mlp",

                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),

                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(
                                    128,
                                    64,
                                ),
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

            # ------------------------------------------------
            # XGBoost
            # ------------------------------------------------

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

        # ====================================================
        # Meta learner
        # ====================================================

        meta_learner = RidgeCV(
            alphas=np.logspace(
                -3,
                3,
                15,
            )
        )

        # ====================================================
        # Internal stacking CV
        # ====================================================

        internal_cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )

        # ====================================================
        # Stacking model
        # ====================================================

        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=internal_cv,
            n_jobs=-1,
        )

    # ========================================================
    # 9. RUN FIVE-FOLD EVALUATION
    # ========================================================

    def run_stacking_pipeline(
        self,
        dataset_csv,
        metric_evaluator: Metric,
        embedding_sources,
        pooling="mean",
    ):

        """
        Run five-fold evaluation with a fixed independent
        test set.

        Dataset design:

            Development = 4,074
            Fixed test = 719

        Outer CV:

            Fold 1:
                Train = 4 development folds
                Validation = 1 development fold
                Test = fixed 719

            ...

            Fold 5:
                Train = 4 development folds
                Validation = 1 development fold
                Test = fixed 719

        The fixed test set is never used during model fitting
        or validation.
        """

        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        if not os.path.exists(dataset_csv):

            raise FileNotFoundError(
                f"Dataset CSV not found: "
                f"{dataset_csv}"
            )

        if not embedding_sources:

            raise ValueError(
                "embedding_sources cannot be empty."
            )

        # ----------------------------------------------------
        # Verify requested embedding sources
        # ----------------------------------------------------

        missing_sources = [
            source
            for source in embedding_sources
            if source not in self.embeddings
        ]

        if missing_sources:

            raise ValueError(
                "Requested embedding sources are not loaded: "
                f"{missing_sources}\n"
                f"Available: {list(self.embeddings.keys())}"
            )

        print("\n" + "=" * 70)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 70)

        print(
            f"Embedding sources: "
            f"{embedding_sources}"
        )

        print(
            f"Pooling: {pooling}"
        )

        print("=" * 70)

        val_metrics = []
        test_metrics = []
        fold_results = []

        fixed_test_sequences = None

        # ====================================================
        # Outer five-fold CV
        # ====================================================

        for fold_number in range(1, 6):

            print("\n" + "=" * 70)
            print(
                f"FOLD {fold_number}/5"
            )
            print("=" * 70)

            # ------------------------------------------------
            # Get split
            # ------------------------------------------------

            train_set, val_set, test_set = (
                data_util.get_cv_split(
                    input_csv_path=dataset_csv,
                    fold=fold_number,
                )
            )

            print(
                f"Train      : {len(train_set)}"
            )

            print(
                f"Validation : {len(val_set)}"
            )

            print(
                f"Fixed Test : {len(test_set)}"
            )

            # ------------------------------------------------
            # Verify fixed test set
            # ------------------------------------------------

            test_sequences = (
                test_set["sequence"]
                .astype(str)
                .tolist()
            )

            if fixed_test_sequences is None:

                fixed_test_sequences = (
                    test_sequences.copy()
                )

            elif sorted(test_sequences) != sorted(
                fixed_test_sequences
            ):

                raise ValueError(
                    "Fixed test set changed across folds. "
                    f"Mismatch detected at fold "
                    f"{fold_number}."
                )

            # ------------------------------------------------
            # Prepare features
            # ------------------------------------------------

            print("\nPreparing training features...")

            X_train, y_train = self.prepare_split(
                train_set,
                embedding_sources=embedding_sources,
                pooling=pooling,
            )

            print("Preparing validation features...")

            X_val, y_val = self.prepare_split(
                val_set,
                embedding_sources=embedding_sources,
                pooling=pooling,
            )

            print("Preparing fixed-test features...")

            X_test, y_test = self.prepare_split(
                test_set,
                embedding_sources=embedding_sources,
                pooling=pooling,
            )

            # ------------------------------------------------
            # Verify dimensions
            # ------------------------------------------------

            feature_dim = X_train.shape[1]

            if X_val.shape[1] != feature_dim:

                raise ValueError(
                    "Training and validation feature "
                    "dimensions do not match."
                )

            if X_test.shape[1] != feature_dim:

                raise ValueError(
                    "Training and test feature "
                    "dimensions do not match."
                )

            print(
                f"\nFinal feature dimension: "
                f"{feature_dim}"
            )

            # ------------------------------------------------
            # Build model
            # ------------------------------------------------

            model = (
                self.build_stacking_ensemble()
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            print(
                "\nTraining stacking ensemble..."
            )

            model.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            val_pred = model.predict(
                X_val
            )

            test_pred = model.predict(
                X_test
            )

            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------

            y_val = np.asarray(
                y_val,
                dtype=np.float64,
            ).reshape(-1)

            y_test = np.asarray(
                y_test,
                dtype=np.float64,
            ).reshape(-1)

            val_pred = np.asarray(
                val_pred,
                dtype=np.float64,
            ).reshape(-1)

            test_pred = np.asarray(
                test_pred,
                dtype=np.float64,
            ).reshape(-1)

            # ------------------------------------------------
            # Prediction length checks
            # ------------------------------------------------

            if len(y_val) != len(val_pred):

                raise ValueError(
                    "Validation target/prediction "
                    "length mismatch."
                )

            if len(y_test) != len(test_pred):

                raise ValueError(
                    "Test target/prediction "
                    "length mismatch."
                )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            val_metric = (
                metric_evaluator.compute_metrics(
                    y_val,
                    val_pred,
                )
            )

            test_metric = (
                metric_evaluator.compute_metrics(
                    y_test,
                    test_pred,
                )
            )

            val_metrics.append(
                val_metric
            )

            test_metrics.append(
                test_metric
            )

            # =================================================
            # Print validation
            # =================================================

            print(
                f"\nFold {fold_number} Validation:"
            )

            self.print_metrics(
                val_metric
            )

            # =================================================
            # Print fixed test
            # =================================================

            print(
                f"\nFold {fold_number} Fixed Test:"
            )

            self.print_metrics(
                test_metric
            )

            # =================================================
            # Analyze meta learner
            # =================================================

            stacking_weights = (
                self.analyze_stacking_weights(
                    model,
                    fold_number,
                )
            )

            # =================================================
            # Store fold results
            # =================================================

            fold_results.append(
                {
                    "fold": fold_number,

                    "val_y_true": y_val,
                    "val_y_pred": val_pred,

                    "test_y_true": y_test,
                    "test_y_pred": test_pred,

                    "stacking_weights":
                        stacking_weights,

                    "model": model,
                }
            )

        # ====================================================
        # Final summary
        # ====================================================

        print("\n")
        print("=" * 70)
        print(
            "FINAL 5-FOLD SUMMARY — FIXED TEST SET"
        )
        print("=" * 70)

        summary_df = (
            self.summarize_metrics(
                test_metrics
            )
        )

        print(
            summary_df.to_string(
                index=False
            )
        )

        # ====================================================
        # Return results
        # ====================================================

        return {
            "validation_metrics":
                val_metrics,

            "test_metrics":
                test_metrics,

            "summary":
                summary_df,

            "fold_results":
                fold_results,
        }

    # ========================================================
    # 10. PRINT METRICS
    # ========================================================

    @staticmethod
    def print_metrics(metrics):

        """
        Print regression metrics in a consistent order.
        """

        ordered_keys = [
            "Pearson_R",
            "Spearman_R",
            "Kendall_Tau",
            "RMSE",
            "MAE",
            "R2",
        ]

        for key in ordered_keys:

            if key in metrics:

                print(
                    f"  {key:<15}: "
                    f"{metrics[key]:.4f}"
                )

    # ========================================================
    # 11. SUMMARIZE METRICS
    # ========================================================

    @staticmethod
    def summarize_metrics(
        metrics_list,
    ):

        """
        Calculate mean and sample standard deviation
        across the five outer folds.
        """

        metric_names = [
            "Pearson_R",
            "Spearman_R",
            "Kendall_Tau",
            "RMSE",
            "MAE",
            "R2",
        ]

        rows = []

        for metric_name in metric_names:

            values = [
                metric[metric_name]
                for metric in metrics_list
                if metric_name in metric
            ]

            if not values:
                continue

            rows.append(
                {
                    "Metric": metric_name,

                    "Mean": np.mean(
                        values
                    ),

                    "Std": np.std(
                        values,
                        ddof=1,
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ========================================================
    # 12. ANALYZE STACKING WEIGHTS
    # ========================================================

    @staticmethod
    def analyze_stacking_weights(
        stacking_model,
        fold_idx,
    ):

        """
        Extract meta-learner coefficients.

        Coefficients represent the relationship between
        base-model predictions and the final RidgeCV
        prediction.

        Absolute coefficient magnitude should NOT be
        interpreted as direct predictive importance or
        percentage contribution.
        """

        estimator_names = [
            name
            for name, _ in stacking_model.estimators
        ]

        meta_learner = (
            stacking_model.final_estimator_
        )

        coefficients = np.asarray(
            meta_learner.coef_
        ).ravel()

        if len(coefficients) != len(
            estimator_names
        ):

            raise ValueError(
                "Number of meta-learner coefficients "
                "does not match number of base "
                "estimators."
            )

        weights_df = pd.DataFrame(
            {
                "Base_Model":
                    estimator_names,

                "Coefficient":
                    coefficients,

                "Absolute_Weight":
                    np.abs(coefficients),
            }
        )

        total = (
            weights_df["Absolute_Weight"]
            .sum()
        )

        if total > 0:

            weights_df[
                "Coefficient_Abs_Percent"
            ] = (
                weights_df[
                    "Absolute_Weight"
                ]
                / total
                * 100
            )

        else:

            weights_df[
                "Coefficient_Abs_Percent"
            ] = 0.0

        weights_df = (
            weights_df
            .sort_values(
                "Absolute_Weight",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        print(
            f"\nMeta-learner weights — "
            f"Fold {fold_idx}:"
        )

        print(
            weights_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Intercept
        # ----------------------------------------------------

        intercept = float(
            np.asarray(
                meta_learner.intercept_
            ).ravel()[0]
        )

        print(
            f"\nMeta learner intercept: "
            f"{intercept:.4f}"
        )

        return weights_df

    # ========================================================
    # 13. SAVE RESULTS
    # ========================================================

    @staticmethod
    def save_results(
        results,
        output_dir,
        prefix="stacking",
    ):

        """
        Save:

            1. Summary metrics
            2. Fold-level metrics
            3. Meta-learner coefficients
        """

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_file = os.path.join(
            output_dir,
            f"{prefix}_summary.csv",
        )

        results[
            "summary"
        ].to_csv(
            summary_file,
            index=False,
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_rows = []

        for fold_idx, metrics in enumerate(
            results["test_metrics"],
            start=1,
        ):

            row = {
                "Fold": fold_idx
            }

            row.update(metrics)

            fold_rows.append(row)

        fold_df = pd.DataFrame(
            fold_rows
        )

        fold_file = os.path.join(
            output_dir,
            f"{prefix}_fold_metrics.csv",
        )

        fold_df.to_csv(
            fold_file,
            index=False,
        )

        # ----------------------------------------------------
        # Meta learner weights
        # ----------------------------------------------------

        weight_rows = []

        for fold_result in results[
            "fold_results"
        ]:

            fold_idx = fold_result[
                "fold"
            ]

            weights_df = (
                fold_result[
                    "stacking_weights"
                ]
                .copy()
            )

            weights_df.insert(
                0,
                "Fold",
                fold_idx,
            )

            weight_rows.append(
                weights_df
            )

        if weight_rows:

            all_weights_df = pd.concat(
                weight_rows,
                ignore_index=True,
            )

            weights_file = os.path.join(
                output_dir,
                f"{prefix}_meta_weights.csv",
            )

            all_weights_df.to_csv(
                weights_file,
                index=False,
            )

            print(
                f"\nSaved meta weights: "
                f"{weights_file}"
            )

        print(
            f"\nSaved summary: "
            f"{summary_file}"
        )

        print(
            f"Saved fold metrics: "
            f"{fold_file}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Dataset
    # ========================================================

    DATASET_CSV = (
        "./data/ecoli_five_fold_dataset.csv"
    )

    # ========================================================
    # Embedding files
    #
    # You can keep all available embeddings here.
    # Only the ones listed in `embedding_sources` are used.
    # ========================================================

    EMBEDDING_FILES = {

        "prott5":
            "./data/prott5/"
            "prot_t5_xl_uniref50_ecoli.npz",

        "ft_prott5":
            "./data/prott5/"
            "ecoli_finetuned_prott5.npz",

        "esm2":
            "./data/esm2/"
            "esm2_t33_650M_UR50D_1280_ecoli.npz",

        # Uncomment when a pre-computed fusion
        # embedding is available.
        #
        # "fusion":
        #     "./data/fusion/"
        #     "fused_embeddings.npz",
    }

    # ========================================================
    # Select experiment
    # ========================================================
    #
    # Examples:
    #
    #   ["prott5"]
    #
    #   ["ft_prott5"]
    #
    #   ["esm2"]
    #
    #   ["prott5", "esm2"]
    #
    #   ["fusion"]
    #
    # ========================================================

    EMBEDDING_SOURCES = [
        "ft_prott5" #prott5
    ]

    # ========================================================
    # Optional handcrafted features
    #
    # Set any path to None to disable that feature.
    # ========================================================

    PHYSIO_PATH = (
        "./data/ecoli_phyiochem.csv"
    )

    BLOSUM_PATH = (
        "./data/ecoli_blosum62_features.csv"
    )

    SINU_PATH = (
        "./data/ecoli_sinusoidal_encoding.csv"
    )

    # ========================================================
    # Output
    # ========================================================

    EXPERIMENT_NAME = (
        "Finetuned_ProtT5"
    )

    OUTPUT_PATH = (
        f"./results/{EXPERIMENT_NAME}"
    )

    # ========================================================
    # Create pipeline
    # ========================================================

    pipeline = StackingRegressionPipeline(

        embedding_files=EMBEDDING_FILES,

        physio_path=PHYSIO_PATH,

        blosum_path=BLOSUM_PATH,

        sinu_path=SINU_PATH,

        random_state=42,
    )

    # ========================================================
    # Metric evaluator
    # ========================================================

    metric = Metric(
        "RegressionMetrics"
    )

    # ========================================================
    # Five-fold evaluation
    # ========================================================

    results = pipeline.run_stacking_pipeline(

        dataset_csv=DATASET_CSV,

        metric_evaluator=metric,

        embedding_sources=EMBEDDING_SOURCES,

        pooling="mean",
    )

    # ========================================================
    # Save results
    # ========================================================

    pipeline.save_results(

        results,

        output_dir=OUTPUT_PATH,

        prefix="stacking",
    )
