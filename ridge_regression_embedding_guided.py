import os
from pathlib import Path

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


class StackingRegressionPipeline:

    # ============================================================
    # 1. INITIALIZATION
    # ============================================================

    def __init__(
        self,
        embedding_file = None,
        physio_path = None,
        blosum_path = None,
        sinu_path = None,
        random_state=42,
    ):
        """
        Embeddings stacking regression pipeline.

        Features:
            residue-level embeddings
            -> mean pooling
            -> stacking regression

        No:
            - physicochemical features
            - BLOSUM features
            - sinusoidal positional encoding
            - CNN features
            - domain-guided attention

        Parameters
        ----------
        prott5_embedding_file : str
            NPZ file containing:

                sequence -> residue-level ProtT5 embedding

        random_state : int
            Random seed.
        """

        self.embedding_file = embedding_file
        self.physio_path = physio_path
        self.blosum_path = blosum_path
        self.sinu_path = sinu_path
        self.random_state = random_state
        

        # --------------------------------------------------------
        # Load ProtT5 embeddings once.
        # --------------------------------------------------------

        self.embeddings = data_util.load_embeddings(embedding_file)
        
        
        if self.physio_path:
            self.physio_features = physiochem_feature_extractor.load_physio_features_as_numpy_all(physio_path)
        
        if self.blosum_path: 
            self.blosum_features = blosum62.load_blosum62_features(blosum_path)
        
        if self.sinu_path:
            self.sinu_features = position_encoding_extractor.load_sinusoidal_encoding(sinu_path)
            
            

    # ============================================================
    # 3. FEATURE POOLING
    # ============================================================

    @staticmethod
    def mean_pool(arr):
        """
        Mean-pool residue-level ProtT5 embeddings.

        Parameters
        ----------
        arr : np.ndarray
            Shape:

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

        # Already pooled.
        if arr.ndim == 1:
            return arr

        if arr.ndim != 2:
            raise ValueError(
                "Expected ProtT5 embedding to be 1D or 2D, "
                f"but got shape {arr.shape}"
            )

        return np.mean(
            arr,
            axis=0,
        ).astype(np.float32)

    # ============================================================
    # 4. FEATURE EXTRACTION
    # ============================================================

    def get_embedding_features(
        self,
        sequence,
        pooling="mean",
    ):
        """
        Extract PURE ProtT5 features for one sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence.

        pooling : str
            Currently only:

                "mean"

        Returns
        -------
        np.ndarray
            1D ProtT5 feature vector.
        """

        sequence = str(sequence)

        if sequence not in self.embeddings:
            raise KeyError(
                "Sequence not found in ProtT5 embedding file:\n"
                f"{sequence}"
            )

        embedding = self.embeddings[
            sequence
        ]

        if pooling == "mean":

            feature = self.mean_pool(
                embedding
            )

        else:

            raise ValueError(
                f"Unknown pooling method: {pooling}. "
                "Currently only 'mean' is supported."
            )

        return feature.astype(
            np.float32
        )

    # ============================================================
    # 5. PREPARE DATASET
    # ============================================================

    def prepare_split(
        self,
        dataset_split,
        pooling="mean",
    ):
        """
        Convert a dataset DataFrame into X and y.

        Required columns:

            sequence
            value

        Parameters
        ----------
        dataset_split : pd.DataFrame
            Train, validation, or test DataFrame.

        pooling : str
            pooling method.

        Returns
        -------
        X : np.ndarray
            Shape:

                (n_samples, 1024)

        y : np.ndarray
            Shape:

                (n_samples,)
        """

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

            sequence = str(
                row.sequence
            )

            features = self.get_embedding_features(
                sequence,
                pooling=pooling,
            )
            
            if self.physio_features:
                physio_feat = self.physio_features[sequence]
                features = np.concatenate(
                    [features, physio_feat],
                    axis=0
                ) 
            
            if self.blosum_features:
                blosum_feat = self.blosum_features[sequence]
                features = np.concatenate([features, blosum_feat], axis=0)
            
            if self.sinu_features:
                sinu_feat = self.sinu_features[sequence]
                features = np.concatenate([features, sinu_feat], axis=0)
                

            X_list.append(
                features
            )

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
        print(f"Final Shape of Input Features: ", X.shape)

        return X, y

    # ============================================================
    # 6. STACKING MODEL
    # ============================================================

    def build_stacking_ensemble(self):
        """
        Build the stacking regression ensemble.

        Scaling is included inside the individual pipelines.

        This is important because StackingRegressor performs
        internal CV on the outer training data. Each scaler is
        therefore fitted only on the corresponding inner-training
        portion.

        Base estimators:

            1. Ridge
            2. SVR
            3. ExtraTrees
            4. HistGradientBoosting
            5. KNN
            6. MLP
            7. XGBoost

        Meta learner:

            RidgeCV
        """

        base_estimators = [

            # ----------------------------------------------------
            # Ridge
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # SVR
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Extra Trees
            # ----------------------------------------------------

            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=16,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),

            # ----------------------------------------------------
            # HistGradientBoosting
            # ----------------------------------------------------

            (
                "hist_gb",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    l2_regularization=1.0,
                    random_state=self.random_state,
                ),
            ),

            # ----------------------------------------------------
            # KNN
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # MLP
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # XGBoost
            # ----------------------------------------------------

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

        # ========================================================
        # Meta learner
        # ========================================================

        meta_learner = RidgeCV(
            alphas=np.logspace(
                -3,
                3,
                15,
            )
        )

        # ========================================================
        # Internal CV for stacking
        # ========================================================

        internal_cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )

        # ========================================================
        # Stacking model
        # ========================================================

        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=internal_cv,
            n_jobs=-1,
        )

        return stacking_model

    # ============================================================
    # 7. RUN FIVE-FOLD EVALUATION
    # ============================================================

    def run_stacking_pipeline(
        self,
        dataset_csv,
        metric_evaluator,
        pooling="mean",
    ):
        """
        Run five-fold evaluation using a FIXED independent
        test set.

        Dataset design:

            Development:
                4,074 samples

            Fixed independent test:
                719 samples

            Development CV:
                5 folds

        For each outer fold:

            Train:
                4 development folds

            Validation:
                1 development fold

            Test:
                same fixed 719 samples

        The fixed test set is never used for training,
        validation, or model selection.
        """

        # --------------------------------------------------------
        # Check dataset file
        # --------------------------------------------------------

        if not os.path.exists(
            dataset_csv
        ):
            raise FileNotFoundError(
                f"Dataset CSV not found: {dataset_csv}"
            )

        val_metrics = []
        test_metrics = []

        fold_results = []

        fixed_test_sequences = None

        # ========================================================
        # Outer five-fold CV
        # ========================================================

        for fold_number in range(
            1,
            6,
        ):

            print("\n" + "=" * 70)
            print(
                f"FOLD {fold_number}/5"
            )
            print("=" * 70)

            # ----------------------------------------------------
            # Retrieve split
            # ----------------------------------------------------

            train_set, val_set, test_set = (
                data_util.get_cv_split(
                    input_csv_path=dataset_csv,
                    fold=fold_number,
                )  #train_set, val_set and test_set are dataframes
            )

            print(
                f"Train     : {len(train_set)}"
            )

            print(
                f"Validation: {len(val_set)}"
            )

            print(
                f"Fixed Test: {len(test_set)}"
            )

            # ----------------------------------------------------
            # Verify that the test set is fixed
            # ----------------------------------------------------

            test_sequences = (
                test_set["sequence"]
                .astype(str)
                .tolist()
            )

            if fixed_test_sequences is None:

                fixed_test_sequences = (
                    test_sequences
                )

            elif sorted(
                test_sequences
            ) != sorted(
                fixed_test_sequences
            ):

                raise ValueError(
                    "Test set is NOT fixed across folds. "
                    f"Mismatch detected at fold "
                    f"{fold_number}."
                )

            # ----------------------------------------------------
            # Prepare ProtT5 features
            # ----------------------------------------------------

            X_train, y_train = (
                self.prepare_split(
                    train_set,
                    pooling=pooling,
                )
            )

            X_val, y_val = (
                self.prepare_split(
                    val_set,
                    pooling=pooling,
                )
            )

            X_test, y_test = (
                self.prepare_split(
                    test_set,
                    pooling=pooling,
                )
            )

            print(
                f"Feature dimension: "
                f"{X_train.shape[1]}"
            )

            # ----------------------------------------------------
            # Verify feature dimensions
            # ----------------------------------------------------

            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError(
                    "Training and validation feature "
                    "dimensions do not match."
                )

            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError(
                    "Training and test feature "
                    "dimensions do not match."
                )

            # ----------------------------------------------------
            # Build stacking model
            # ----------------------------------------------------

            model = (
                self.build_stacking_ensemble()
            )

            print(
                "\nTraining stacking ensemble..."
            )

            # ----------------------------------------------------
            # Train
            #
            # IMPORTANT:
            # No global StandardScaler is applied here.
            #
            # Scaling is handled inside the individual
            # base-estimator pipelines.
            # ----------------------------------------------------

            model.fit(
                X_train,
                y_train,
            )

            # ----------------------------------------------------
            # Predictions
            # ----------------------------------------------------

            val_pred = model.predict(
                X_val
            )

            test_pred = model.predict(
                X_test
            )

            # ----------------------------------------------------
            # Flatten arrays
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Check prediction lengths
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Compute metrics
            # ----------------------------------------------------

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

            # ====================================================
            # Print validation results
            # ====================================================

            print(
                f"\nFold {fold_number} Validation:"
            )

            self.print_metrics(
                val_metric
            )

            # ====================================================
            # Print fixed-test results
            # ====================================================

            print(
                f"\nFold {fold_number} Fixed Test:"
            )

            self.print_metrics(
                test_metric
            )

            # ====================================================
            # Analyze meta learner
            # ====================================================

            stacking_weights = (
                self.analyze_stacking_weights(
                    model,
                    fold_number,
                )
            )

            # ====================================================
            # Save fold results
            # ====================================================

            fold_result = {

                "fold": fold_number,

                "val_y_true": y_val,
                "val_y_pred": val_pred,

                "test_y_true": y_test,
                "test_y_pred": test_pred,

                "stacking_weights": (
                    stacking_weights
                ),

                "model": model,
            }

            fold_results.append(
                fold_result
            )

        # ========================================================
        # Final summary
        # ========================================================

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

        # ========================================================
        # Return all results
        # ========================================================

        return {

            "validation_metrics": (
                val_metrics
            ),

            "test_metrics": (
                test_metrics
            ),

            "summary": summary_df,

            "fold_results": (
                fold_results
            ),
        }

    # ============================================================
    # 8. PRINT METRICS
    # ============================================================

    @staticmethod
    def print_metrics(
        metrics
    ):
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

    # ============================================================
    # 9. SUMMARIZE FIVE FOLDS
    # ============================================================

    @staticmethod
    def summarize_metrics(
        metrics_list
    ):
        """
        Calculate mean and sample standard deviation
        across the five outer folds.
        """

        rows = []

        metric_names = [
            "Pearson_R",
            "Spearman_R",
            "Kendall_Tau",
            "RMSE",
            "MAE",
            "R2",
        ]

        for key in metric_names:

            values = [
                metric[key]
                for metric in metrics_list
                if key in metric
            ]

            if not values:
                continue

            rows.append(
                {
                    "Metric": key,
                    "Mean": np.mean(
                        values
                    ),
                    "Std": np.std(
                        values,
                        ddof=1,
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )

    # ============================================================
    # 10. META LEARNER ANALYSIS
    # ============================================================

    @staticmethod
    def analyze_stacking_weights(
        stacking_model,
        fold_idx,
    ):
        """
        Extract meta-learner coefficients.

        The coefficients correspond to the base-model
        predictions used by the RidgeCV meta learner.

        Note:
            Absolute coefficient magnitude is useful for
            inspecting the meta learner, but should not be
            interpreted as a direct percentage of predictive
            importance.
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
                "does not match number of base estimators."
            )

        df = pd.DataFrame(
            {
                "Base_Model": estimator_names,

                "Coefficient": (
                    coefficients
                ),

                "Absolute_Weight": np.abs(
                    coefficients
                ),
            }
        )

        total = df[
            "Absolute_Weight"
        ].sum()

        if total > 0:

            df[
                "Coefficient_Abs_Percent"
            ] = (
                df["Absolute_Weight"]
                / total
                * 100
            )

        else:

            df[
                "Coefficient_Abs_Percent"
            ] = 0.0

        df = df.sort_values(
            "Absolute_Weight",
            ascending=False,
        )

        print(
            f"\nMeta-learner weights — "
            f"Fold {fold_idx}:"
        )

        print(
            df.to_string(
                index=False
            )
        )

        # --------------------------------------------------------
        # Intercept
        # --------------------------------------------------------

        intercept = float(
            np.asarray(
                meta_learner.intercept_
            ).ravel()[0]
        )

        print(
            f"\nMeta learner intercept: "
            f"{intercept:.4f}"
        )

        return df

    # ============================================================
    # 11. SAVE RESULTS
    # ============================================================

    @staticmethod
    def save_results(
        results,
        output_dir,
        prefix="prott5_stacking",
    ):
        """
        Save summary and fold-level test metrics.
        """

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # ========================================================
        # Summary
        # ========================================================

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

        # ========================================================
        # Fold-level metrics
        # ========================================================

        fold_rows = []

        for fold_idx, metrics in enumerate(
            results["test_metrics"],
            start=1,
        ):

            row = {
                "Fold": fold_idx
            }

            row.update(
                metrics
            )

            fold_rows.append(
                row
            )

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

        # ========================================================
        # Save stacking coefficients
        # ========================================================

        weight_rows = []

        for fold_result in results[
            "fold_results"
        ]:

            fold_idx = (
                fold_result["fold"]
            )

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
                f"Saved meta weights: "
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


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":
    
    DATASET_CSV = (
        "./data/ecoli_five_fold_dataset.csv"
    )
    EMBEDDING_FILES = {
        'PROTT5':'prott5/prot_t5_xl_uniref50_ecoli.npz',
        'PROSTT5': 'prott5/ProstT5_ecoli.npz' ,
        'ESM_320': 'esm2/esm2_t6_8M_UR50D_dim_320_ecoli.npz',
        'ESM_640': 'esm2/esm2_t30_150M_UR50D_dim_640_ecoli.npz',
        'ESM_1280': 'esm2/esm2_t33_650M_UR50D_1280_ecoli.npz'
    }


    # ============================================================
    # Embedding based stacking pipeline
    # ============================================================

    embedding_path = f"./data/{EMBEDDING_FILES['PROTT5']}"
    print("Embedding Path Set : ", embedding_path)
    
    out_name = Path(embedding_path).name.split(".")[0]
    output_path = f"./results/{out_name}_emb_physio_blosum_Sinu"
    print(f"Output Path: {output_path}")

    
    pipeline = (
        StackingRegressionPipeline(
            embedding_file=embedding_path,
            physio_path = "./data/ecoli_phyiochem.csv",
            blosum_path = "./data/ecoli_blosum62_features.csv",
            sinu_path = "./data/ecoli_sinusoidal_encoding.csv",
            random_state=42,
        )
    )
    
    metric = Metric(
        "RegressionMetrics"
    )
    
    # ============================================================
    # Five-fold evaluation
    # ============================================================

    results = (
        pipeline.run_stacking_pipeline(
            dataset_csv=DATASET_CSV,
            metric_evaluator=metric,
            pooling="mean",
        )
    )

    # ============================================================
    # Save results
    # ============================================================
    
    
    pipeline.save_results(
        results,
        output_dir=output_path,
        prefix="",
    )