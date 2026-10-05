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


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42
N_FOLDS = 5

# ------------------------------------------------
# Hemolysis dataset columns
# ------------------------------------------------

SEQUENCE_COLUMN = "SEQUENCE"

# IMPORTANT:
# Set this to the actual hemolysis target column.
#
# Example:
# TARGET_COLUMN = "HC50"
#
# The script will show available columns if this is incorrect.
TARGET_COLUMN = "μM" #HC50

# Optional label column.
# It is NOT used for regression.
LABEL_COLUMN = "label"


# ================================================================
# HEMOLYSIS FEATURE LOADERS
# ================================================================

def load_hemolysis_prott5_embeddings(npz_path):
    """
    Load hemolysis ProtT5 residue-level embeddings.

    Expected NPZ fields:
        original_sequences
        processed_sequences
        embeddings

    Returns:
        dict:
            sequence -> residue-level embedding array
    """

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"ProtT5 embedding file not found:\n{npz_path}"
        )

    data = np.load(
        npz_path,
        allow_pickle=True
    )

    required_fields = [
        "original_sequences",
        "embeddings",
    ]

    for field in required_fields:

        if field not in data.files:
            raise ValueError(
                f"Required field '{field}' not found in:\n"
                f"{npz_path}\n"
                f"Available fields: {data.files}"
            )

    sequences = data["original_sequences"]
    embeddings = data["embeddings"]

    if len(sequences) != len(embeddings):
        raise ValueError(
            "Mismatch between number of sequences and "
            "number of ProtT5 embeddings."
        )

    embedding_dict = {}

    for sequence, embedding in zip(
        sequences,
        embeddings
    ):

        sequence = str(sequence).strip()

        embedding_dict[sequence] = embedding

    print(
        f"Loaded ProtT5 embeddings: "
        f"{len(embedding_dict)} sequences"
    )

    if embedding_dict:

        first_sequence = next(
            iter(embedding_dict)
        )

        print(
            f"First ProtT5 embedding shape: "
            f"{np.asarray(embedding_dict[first_sequence]).shape}"
        )

    return embedding_dict


def load_hemolysis_physio_features(csv_path):
    """
    Load hemolysis physicochemical features.

    Expected format:

        SEQUENCE, feature_1, feature_2, ...

    Returns:
        dict:
            sequence -> feature vector
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Physicochemical feature file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if SEQUENCE_COLUMN not in df.columns:

        raise ValueError(
            f"'{SEQUENCE_COLUMN}' column not found in:\n"
            f"{csv_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    feature_columns = [
        col
        for col in df.columns
        if col != SEQUENCE_COLUMN
    ]

    if not feature_columns:

        raise ValueError(
            f"No physicochemical feature columns found in:\n"
            f"{csv_path}"
        )

    sequences = (
        df[SEQUENCE_COLUMN]
        .astype(str)
        .str.strip()
        .tolist()
    )

    features = (
        df[feature_columns]
        .values
        .astype(np.float32)
    )

    feature_dict = {
        sequence: features[i]
        for i, sequence in enumerate(sequences)
    }

    print(
        f"Loaded physicochemical features: "
        f"{features.shape}"
    )

    return feature_dict


def load_hemolysis_blosum62_features(csv_path):
    """
    Load hemolysis BLOSUM62 features.

    Expected columns:

        SEQUENCE
        blosum62_A
        blosum62_R
        ...
        blosum62_V

    Returns:
        dict:
            sequence -> 20-dimensional feature vector
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"BLOSUM62 feature file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if SEQUENCE_COLUMN not in df.columns:

        raise ValueError(
            f"'{SEQUENCE_COLUMN}' column not found in:\n"
            f"{csv_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    canonical_aas = list(
        "ARNDCQEGHILKMFPSTWYV"
    )

    blosum_columns = [
        f"blosum62_{aa}"
        for aa in canonical_aas
    ]

    missing_columns = [
        col
        for col in blosum_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing BLOSUM62 columns:\n"
            f"{missing_columns}"
        )

    sequences = (
        df[SEQUENCE_COLUMN]
        .astype(str)
        .str.strip()
        .tolist()
    )

    features = (
        df[blosum_columns]
        .values
        .astype(np.float32)
    )

    feature_dict = {
        sequence: features[i]
        for i, sequence in enumerate(sequences)
    }

    print(
        f"Loaded BLOSUM62 features: "
        f"{features.shape}"
    )

    return feature_dict


def load_hemolysis_sinusoidal_features(csv_path):
    """
    Load global sinusoidal positional encodings.

    Expected columns:

        SEQUENCE
        pe_0
        pe_1
        ...
        pe_D

    Returns:
        dict:
            sequence -> positional feature vector
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Sinusoidal feature file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if SEQUENCE_COLUMN not in df.columns:

        raise ValueError(
            f"'{SEQUENCE_COLUMN}' column not found in:\n"
            f"{csv_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    pe_columns = [
        col
        for col in df.columns
        if col.startswith("pe_")
    ]

    if not pe_columns:

        raise ValueError(
            f"No sinusoidal positional encoding columns "
            f"found in:\n{csv_path}"
        )

    sequences = (
        df[SEQUENCE_COLUMN]
        .astype(str)
        .str.strip()
        .tolist()
    )

    features = (
        df[pe_columns]
        .values
        .astype(np.float32)
    )

    feature_dict = {
        sequence: features[i]
        for i, sequence in enumerate(sequences)
    }

    print(
        f"Loaded sinusoidal features: "
        f"{features.shape}"
    )

    return feature_dict


# ================================================================
# STACKING REGRESSION PIPELINE
# ================================================================

class StackingRegressionPipeline:

    # ============================================================
    # 1. INITIALIZATION
    # ============================================================

    def __init__(
        self,
        prott5_path,
        physio_path=None,
        blosum_path=None,
        sinu_path=None,
        random_state=42,
    ):
        """
        Hemolysis peptide regression pipeline.

        Input features:

            1. ProtT5 residue-level embeddings
               -> mean pooling

            2. Physicochemical features

            3. BLOSUM62 features

            4. Sinusoidal positional encoding

        Regression target:

            log10(hemolysis concentration)

        Base regressors:

            Ridge
            SVR
            ExtraTrees
            HistGradientBoosting
            KNN
            MLP
            XGBoost

        Meta learner:

            RidgeCV
        """

        self.random_state = random_state

        # --------------------------------------------------------
        # ProtT5
        # --------------------------------------------------------

        print("\n" + "=" * 70)
        print("LOADING HEMOLYSIS FEATURE REPRESENTATIONS")
        print("=" * 70)

        print("\nLoading ProtT5 embeddings...")

        self.prott5_embeddings = (
            load_hemolysis_prott5_embeddings(
                prott5_path
            )
        )

        # --------------------------------------------------------
        # Physicochemical
        # --------------------------------------------------------

        self.physio_features = None

        if physio_path is not None:

            print(
                "\nLoading physicochemical features..."
            )

            self.physio_features = (
                load_hemolysis_physio_features(
                    physio_path
                )
            )

        # --------------------------------------------------------
        # BLOSUM62
        # --------------------------------------------------------

        self.blosum_features = None

        if blosum_path is not None:

            print(
                "\nLoading BLOSUM62 features..."
            )

            self.blosum_features = (
                load_hemolysis_blosum62_features(
                    blosum_path
                )
            )

        # --------------------------------------------------------
        # Sinusoidal positional encoding
        # --------------------------------------------------------

        self.sinu_features = None

        if sinu_path is not None:

            print(
                "\nLoading sinusoidal positional features..."
            )

            self.sinu_features = (
                load_hemolysis_sinusoidal_features(
                    sinu_path
                )
            )

        print("\n" + "=" * 70)
        print("FEATURE LOADING COMPLETED")
        print("=" * 70)

    # ============================================================
    # 2. MEAN POOLING
    # ============================================================

    @staticmethod
    def mean_pool(arr):
        """
        Mean-pool residue-level ProtT5 embeddings.

        Input:
            (sequence_length, embedding_dimension)

        Output:
            (embedding_dimension,)
        """

        arr = np.asarray(
            arr,
            dtype=np.float32,
        )

        # Already pooled
        if arr.ndim == 1:

            return arr

        if arr.ndim != 2:

            raise ValueError(
                "Expected ProtT5 embedding to be "
                "1D or 2D, but received shape "
                f"{arr.shape}"
            )

        return np.mean(
            arr,
            axis=0,
        ).astype(np.float32)

    # ============================================================
    # 3. PROTT5 FEATURE EXTRACTION
    # ============================================================

    def get_prott5_features(
        self,
        sequence,
        pooling="mean",
    ):
        """
        Extract mean-pooled ProtT5 representation.
        """

        sequence = str(sequence).strip()

        if sequence not in self.prott5_embeddings:

            raise KeyError(
                "Sequence not found in ProtT5 embedding file:\n"
                f"{sequence}"
            )

        embedding = (
            self.prott5_embeddings[
                sequence
            ]
        )

        if pooling == "mean":

            return self.mean_pool(
                embedding
            )

        raise ValueError(
            f"Unknown pooling method: {pooling}"
        )

    # ============================================================
    # 4. PREPARE DATASET
    # ============================================================

    def prepare_split(
        self,
        dataset,
        pooling="mean",
    ):
        """
        Convert a dataframe into:

            X = heterogeneous feature matrix

            y = log10-transformed hemolysis target

        Required columns:

            SEQUENCE
            TARGET_COLUMN

        The optional label column is not used.
        """

        if not isinstance(
            dataset,
            pd.DataFrame,
        ):

            raise TypeError(
                "dataset must be a pandas DataFrame."
            )

        required_columns = {
            SEQUENCE_COLUMN,
            TARGET_COLUMN,
        }

        missing_columns = (
            required_columns
            - set(dataset.columns)
        )

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}\n"
                f"Available columns: "
                f"{list(dataset.columns)}"
            )

        X_list = []
        y_list = []

        for row in dataset.itertuples(
            index=False
        ):

            # ----------------------------------------------------
            # Sequence
            # ----------------------------------------------------

            sequence = str(
                getattr(
                    row,
                    SEQUENCE_COLUMN
                )
            ).strip().upper()

            # ----------------------------------------------------
            # Feature collection
            # ----------------------------------------------------

            features = []

            # ----------------------------------------------------
            # ProtT5
            # ----------------------------------------------------

            protT5_feature = (
                self.get_prott5_features(
                    sequence,
                    pooling=pooling,
                )
            )

            features.append(
                protT5_feature
            )

            # ----------------------------------------------------
            # Physicochemical
            # ----------------------------------------------------

            if self.physio_features is not None:

                if sequence not in self.physio_features:

                    raise KeyError(
                        "Sequence not found in "
                        "physicochemical feature file:\n"
                        f"{sequence}"
                    )

                physio_feat = np.asarray(
                    self.physio_features[
                        sequence
                    ],
                    dtype=np.float32,
                )

                features.append(
                    physio_feat
                )

            # ----------------------------------------------------
            # BLOSUM62
            # ----------------------------------------------------

            if self.blosum_features is not None:

                if sequence not in self.blosum_features:

                    raise KeyError(
                        "Sequence not found in "
                        "BLOSUM62 feature file:\n"
                        f"{sequence}"
                    )

                blosum_feat = np.asarray(
                    self.blosum_features[
                        sequence
                    ],
                    dtype=np.float32,
                )

                features.append(
                    blosum_feat
                )

            # ----------------------------------------------------
            # Sinusoidal positional encoding
            # ----------------------------------------------------

            if self.sinu_features is not None:

                if sequence not in self.sinu_features:

                    raise KeyError(
                        "Sequence not found in "
                        "sinusoidal encoding file:\n"
                        f"{sequence}"
                    )

                sinu_feat = np.asarray(
                    self.sinu_features[
                        sequence
                    ],
                    dtype=np.float32,
                )

                features.append(
                    sinu_feat
                )

            # ----------------------------------------------------
            # Concatenate representations
            # ----------------------------------------------------

            features = np.concatenate(
                features,
                axis=0,
            )

            X_list.append(
                features
            )

            # ----------------------------------------------------
            # Target
            # ----------------------------------------------------

            target_value = float(
                getattr(
                    row,
                    TARGET_COLUMN
                )
            )

            if not np.isfinite(
                target_value
            ):

                raise ValueError(
                    f"Non-finite target value "
                    f"for sequence {sequence}: "
                    f"{target_value}"
                )

            if target_value <= 0:

                raise ValueError(
                    f"Target must be > 0 for "
                    f"log10 transformation. "
                    f"Found {target_value} "
                    f"for sequence {sequence}"
                )

            log_target = np.log10(
                target_value
            )

            y_list.append(
                log_target
            )

        if not X_list:

            raise ValueError(
                "Dataset is empty."
            )

        X = np.asarray(
            X_list,
            dtype=np.float32,
        )

        y = np.asarray(
            y_list,
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(X)
        ):

            raise ValueError(
                "Feature matrix contains "
                "NaN or infinite values."
            )

        print(
            f"Feature matrix shape: {X.shape}"
        )

        print(
            f"Target shape: {y.shape}"
        )

        return X, y

    # ============================================================
    # 5. BUILD STACKING ENSEMBLE
    # ============================================================

    def build_stacking_ensemble(self):

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

        # --------------------------------------------------------
        # Meta learner
        # --------------------------------------------------------

        meta_learner = RidgeCV(
            alphas=np.logspace(
                -3,
                3,
                15,
            )
        )

        # --------------------------------------------------------
        # Internal CV
        # --------------------------------------------------------

        internal_cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )

        # --------------------------------------------------------
        # Stacking model
        # --------------------------------------------------------

        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=internal_cv,
            n_jobs=-1,
        )

        return stacking_model

    # ============================================================
    # 6. FIVE-FOLD CV + INDEPENDENT TEST
    # ============================================================

    def run_stacking_pipeline(
        self,
        cross_val_csv,
        independent_csv,
        metric_evaluator,
        pooling="mean",
    ):
        """
        Five-fold CV on the hemolysis development dataset.

        Independent dataset is never used during training.

        For every outer fold:

            4 folds -> training
            1 fold  -> validation
            independent dataset -> test
        """

        # ========================================================
        # Check files
        # ========================================================

        if not os.path.exists(
            cross_val_csv
        ):

            raise FileNotFoundError(
                f"Cross-validation dataset not found:\n"
                f"{cross_val_csv}"
            )

        if not os.path.exists(
            independent_csv
        ):

            raise FileNotFoundError(
                f"Independent dataset not found:\n"
                f"{independent_csv}"
            )

        # ========================================================
        # Load datasets
        # ========================================================

        cv_dataset = pd.read_csv(
            cross_val_csv
        )

        independent_dataset = pd.read_csv(
            independent_csv
        )

        print("\n" + "=" * 70)
        print("HEMOLYSIS DATASET INFORMATION")
        print("=" * 70)

        print(
            f"Cross-validation dataset : "
            f"{len(cv_dataset)} samples"
        )

        print(
            f"Independent dataset      : "
            f"{len(independent_dataset)} samples"
        )

        print(
            f"Target column            : "
            f"{TARGET_COLUMN}"
        )

        # ========================================================
        # Validate columns
        # ========================================================

        required_columns = {
            SEQUENCE_COLUMN,
            TARGET_COLUMN,
        }

        for name, dataset in [
            (
                "cross-validation",
                cv_dataset
            ),
            (
                "independent",
                independent_dataset
            ),
        ]:

            missing = (
                required_columns
                - set(dataset.columns)
            )

            if missing:

                raise ValueError(
                    f"{name} dataset is missing columns: "
                    f"{sorted(missing)}\n"
                    f"Available columns: "
                    f"{list(dataset.columns)}"
                )

        # ========================================================
        # Clean sequence strings
        # ========================================================

        cv_dataset = cv_dataset.copy()

        independent_dataset = (
            independent_dataset.copy()
        )

        cv_dataset[
            SEQUENCE_COLUMN
        ] = (
            cv_dataset[
                SEQUENCE_COLUMN
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        independent_dataset[
            SEQUENCE_COLUMN
        ] = (
            independent_dataset[
                SEQUENCE_COLUMN
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # ========================================================
        # Check duplicate sequences
        # ========================================================

        cv_duplicates = (
            cv_dataset[
                SEQUENCE_COLUMN
            ]
            .duplicated()
            .sum()
        )

        independent_duplicates = (
            independent_dataset[
                SEQUENCE_COLUMN
            ]
            .duplicated()
            .sum()
        )

        if cv_duplicates > 0:

            print(
                f"\nWARNING: Cross-validation dataset "
                f"contains {cv_duplicates} duplicate sequences."
            )

        if independent_duplicates > 0:

            print(
                f"WARNING: Independent dataset "
                f"contains {independent_duplicates} duplicate sequences."
            )

        # ========================================================
        # Check train/test sequence overlap
        # ========================================================

        cv_sequences = set(
            cv_dataset[
                SEQUENCE_COLUMN
            ]
        )

        independent_sequences = set(
            independent_dataset[
                SEQUENCE_COLUMN
            ]
        )

        overlap = (
            cv_sequences
            &
            independent_sequences
        )

        if overlap:

            raise ValueError(
                "Data leakage detected: "
                f"{len(overlap)} sequences occur in both "
                "cross-validation and independent datasets."
            )

        print(
            "\nNo sequence overlap detected between "
            "development and independent datasets."
        )

        # ========================================================
        # Prepare independent test features ONCE
        # ========================================================

        print(
            "\nPreparing independent dataset..."
        )

        X_independent, y_independent = (
            self.prepare_split(
                independent_dataset,
                pooling=pooling,
            )
        )

        print(
            f"Independent feature shape: "
            f"{X_independent.shape}"
        )

        # ========================================================
        # Outer five-fold CV
        # ========================================================

        outer_cv = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=self.random_state,
        )

        validation_metrics = []

        independent_test_metrics = []

        fold_results = []

        # ========================================================
        # Run folds
        # ========================================================

        for fold_number, (
            train_indices,
            val_indices,
        ) in enumerate(
            outer_cv.split(cv_dataset),
            start=1,
        ):

            print("\n")
            print("=" * 70)
            print(
                f"FOLD {fold_number}/{N_FOLDS}"
            )
            print("=" * 70)

            # ----------------------------------------------------
            # Split development dataset
            # ----------------------------------------------------

            train_set = (
                cv_dataset
                .iloc[train_indices]
                .reset_index(drop=True)
            )

            val_set = (
                cv_dataset
                .iloc[val_indices]
                .reset_index(drop=True)
            )

            print(
                f"Training        : {len(train_set)}"
            )

            print(
                f"Validation      : {len(val_set)}"
            )

            print(
                f"Independent test: "
                f"{len(independent_dataset)}"
            )

            # ----------------------------------------------------
            # Training features
            # ----------------------------------------------------

            print(
                "\nPreparing training features..."
            )

            X_train, y_train = (
                self.prepare_split(
                    train_set,
                    pooling=pooling,
                )
            )

            # ----------------------------------------------------
            # Validation features
            # ----------------------------------------------------

            print(
                "\nPreparing validation features..."
            )

            X_val, y_val = (
                self.prepare_split(
                    val_set,
                    pooling=pooling,
                )
            )

            # ----------------------------------------------------
            # Feature dimension checks
            # ----------------------------------------------------

            if (
                X_train.shape[1]
                !=
                X_val.shape[1]
            ):

                raise ValueError(
                    "Training and validation feature "
                    "dimensions do not match."
                )

            if (
                X_train.shape[1]
                !=
                X_independent.shape[1]
            ):

                raise ValueError(
                    "Training and independent-test feature "
                    "dimensions do not match."
                )

            print(
                f"\nFeature dimension: "
                f"{X_train.shape[1]}"
            )

            # ----------------------------------------------------
            # Build model
            # ----------------------------------------------------

            model = (
                self.build_stacking_ensemble()
            )

            # ----------------------------------------------------
            # Train
            # ----------------------------------------------------

            print(
                "\nTraining stacking ensemble..."
            )

            model.fit(
                X_train,
                y_train,
            )

            print(
                "Training completed."
            )

            # ----------------------------------------------------
            # Validation prediction
            # ----------------------------------------------------

            val_pred = model.predict(
                X_val
            )

            # ----------------------------------------------------
            # Independent prediction
            # ----------------------------------------------------

            independent_pred = model.predict(
                X_independent
            )

            # ----------------------------------------------------
            # Convert arrays
            # ----------------------------------------------------

            y_val = np.asarray(
                y_val,
                dtype=np.float64,
            ).reshape(-1)

            val_pred = np.asarray(
                val_pred,
                dtype=np.float64,
            ).reshape(-1)

            y_independent = np.asarray(
                y_independent,
                dtype=np.float64,
            ).reshape(-1)

            independent_pred = np.asarray(
                independent_pred,
                dtype=np.float64,
            ).reshape(-1)

            # ----------------------------------------------------
            # Length checks
            # ----------------------------------------------------

            if len(y_val) != len(val_pred):

                raise ValueError(
                    "Validation target/prediction "
                    "length mismatch."
                )

            if (
                len(y_independent)
                !=
                len(independent_pred)
            ):

                raise ValueError(
                    "Independent target/prediction "
                    "length mismatch."
                )

            # ====================================================
            # Metrics
            # ====================================================

            val_metric = (
                metric_evaluator.compute_metrics(
                    y_val,
                    val_pred,
                )
            )

            independent_metric = (
                metric_evaluator.compute_metrics(
                    y_independent,
                    independent_pred,
                )
            )

            validation_metrics.append(
                val_metric
            )

            independent_test_metrics.append(
                independent_metric
            )

            # ====================================================
            # Validation results
            # ====================================================

            print(
                f"\nFold {fold_number} "
                f"Validation Results "
                f"(log10 target):"
            )

            self.print_metrics(
                val_metric
            )

            # ====================================================
            # Independent results
            # ====================================================

            print(
                f"\nFold {fold_number} "
                f"Independent Test Results "
                f"(log10 target):"
            )

            self.print_metrics(
                independent_metric
            )

            # ====================================================
            # Meta learner analysis
            # ====================================================

            stacking_weights = (
                self.analyze_stacking_weights(
                    model,
                    fold_number,
                )
            )

            # ====================================================
            # Store fold results
            # ====================================================

            fold_result = {

                "fold": fold_number,

                "val_y_true": y_val,

                "val_y_pred": val_pred,

                "independent_y_true":
                    y_independent,

                "independent_y_pred":
                    independent_pred,

                "stacking_weights":
                    stacking_weights,

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
            "FINAL 5-FOLD HEMOLYSIS SUMMARY"
        )
        print("=" * 70)

        print(
            "\nValidation performance:"
        )

        validation_summary = (
            self.summarize_metrics(
                validation_metrics
            )
        )

        print(
            validation_summary.to_string(
                index=False
            )
        )

        print(
            "\nIndependent-test performance:"
        )

        independent_summary = (
            self.summarize_metrics(
                independent_test_metrics
            )
        )

        print(
            independent_summary.to_string(
                index=False
            )
        )

        return {

            "validation_metrics":
                validation_metrics,

            "independent_test_metrics":
                independent_test_metrics,

            "validation_summary":
                validation_summary,

            "independent_test_summary":
                independent_summary,

            "fold_results":
                fold_results,
        }

    # ============================================================
    # 7. PRINT METRICS
    # ============================================================

    @staticmethod
    def print_metrics(metrics):

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
    # 8. SUMMARIZE METRICS
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
    # 9. META-LEARNER ANALYSIS
    # ============================================================

    @staticmethod
    def analyze_stacking_weights(
        stacking_model,
        fold_idx,
    ):
        """
        Extract coefficients of the RidgeCV meta learner.
        """

        estimator_names = [
            name
            for name, _
            in stacking_model.estimators
        ]

        meta_learner = (
            stacking_model.final_estimator_
        )

        coefficients = np.asarray(
            meta_learner.coef_
        ).ravel()

        if (
            len(coefficients)
            !=
            len(estimator_names)
        ):

            raise ValueError(
                "Number of meta-learner coefficients "
                "does not match number of base estimators."
            )

        df = pd.DataFrame(
            {
                "Base_Model":
                    estimator_names,

                "Coefficient":
                    coefficients,

                "Absolute_Weight":
                    np.abs(
                        coefficients
                    ),
            }
        )

        total = (
            df["Absolute_Weight"]
            .sum()
        )

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
    # 10. SAVE RESULTS
    # ============================================================

    @staticmethod
    def save_results(
        results,
        output_dir,
        prefix="PepMetaStack_hemolysis",
    ):

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Validation summary
        # --------------------------------------------------------

        validation_summary_file = os.path.join(
            output_dir,
            f"{prefix}_validation_summary.csv",
        )

        results[
            "validation_summary"
        ].to_csv(
            validation_summary_file,
            index=False,
        )

        # --------------------------------------------------------
        # Independent summary
        # --------------------------------------------------------

        independent_summary_file = os.path.join(
            output_dir,
            f"{prefix}_independent_test_summary.csv",
        )

        results[
            "independent_test_summary"
        ].to_csv(
            independent_summary_file,
            index=False,
        )

        # --------------------------------------------------------
        # Fold-level validation metrics
        # --------------------------------------------------------

        validation_rows = []

        for fold_idx, metrics in enumerate(
            results[
                "validation_metrics"
            ],
            start=1,
        ):

            row = {
                "Fold": fold_idx
            }

            row.update(
                metrics
            )

            validation_rows.append(
                row
            )

        validation_df = pd.DataFrame(
            validation_rows
        )

        validation_file = os.path.join(
            output_dir,
            f"{prefix}_validation_fold_metrics.csv",
        )

        validation_df.to_csv(
            validation_file,
            index=False,
        )

        # --------------------------------------------------------
        # Fold-level independent metrics
        # --------------------------------------------------------

        test_rows = []

        for fold_idx, metrics in enumerate(
            results[
                "independent_test_metrics"
            ],
            start=1,
        ):

            row = {
                "Fold": fold_idx
            }

            row.update(
                metrics
            )

            test_rows.append(
                row
            )

        test_df = pd.DataFrame(
            test_rows
        )

        test_file = os.path.join(
            output_dir,
            f"{prefix}_independent_test_fold_metrics.csv",
        )

        test_df.to_csv(
            test_file,
            index=False,
        )

        # --------------------------------------------------------
        # Stacking coefficients
        # --------------------------------------------------------

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
                f"\nSaved meta weights: "
                f"{weights_file}"
            )

        print(
            f"\nSaved validation summary: "
            f"{validation_summary_file}"
        )

        print(
            f"Saved independent-test summary: "
            f"{independent_summary_file}"
        )

        print(
            f"Saved validation fold metrics: "
            f"{validation_file}"
        )

        print(
            f"Saved independent-test metrics: "
            f"{test_file}"
        )


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":

    # ============================================================
    # Hemolysis datasets
    # ============================================================

    CROSS_VAL_DATASET = (
        "./data/hemolysis/cross_val_dataset.csv"
    )

    INDEPENDENT_DATASET = (
        "./data/hemolysis/independent_dataset.csv"
    )

    # ============================================================
    # Hemolysis feature files
    # ============================================================

    PROTT5_PATH = (
        "./data/hemolysis/hemolysis_prott5.npz"
    )

    PHYSIO_PATH = (
        "./data/hemolysis/hemolysis_physio.csv"
    )

    BLOSUM_PATH = (
        "./data/hemolysis/hemolysis_blosum62_features.csv"
    )

    SINU_PATH = (
        "./data/hemolysis/hemolysis_sinusoidal_encoding.csv"
    )

    # ============================================================
    # Output
    # ============================================================

    OUTPUT_PATH = (
        "./results/PepMetaStack_hemolysis"
    )

    # ============================================================
    # Verify input files before loading
    # ============================================================

    input_files = {
        "Cross-validation dataset":
            CROSS_VAL_DATASET,

        "Independent dataset":
            INDEPENDENT_DATASET,

        "ProtT5 embeddings":
            PROTT5_PATH,

        "Physicochemical features":
            PHYSIO_PATH,

        "BLOSUM62 features":
            BLOSUM_PATH,

        "Sinusoidal features":
            SINU_PATH,
    }

    print("\n" + "=" * 70)
    print("CHECKING HEMOLYSIS INPUT FILES")
    print("=" * 70)

    missing_files = []

    for name, path in input_files.items():

        exists = os.path.exists(path)

        print(
            f"{name:<30}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            missing_files.append(path)

    if missing_files:

        raise FileNotFoundError(
            "\nThe following required files are missing:\n"
            +
            "\n".join(
                f"  {path}"
                for path in missing_files
            )
        )

    # ============================================================
    # Inspect target column before starting
    # ============================================================

    cv_preview = pd.read_csv(
        CROSS_VAL_DATASET,
        nrows=5,
    )

    independent_preview = pd.read_csv(
        INDEPENDENT_DATASET,
        nrows=5,
    )

    print("\n" + "=" * 70)
    print("HEMOLYSIS DATASET COLUMNS")
    print("=" * 70)

    print(
        "\nCross-validation columns:"
    )

    print(
        cv_preview.columns.tolist()
    )

    print(
        "\nIndependent-test columns:"
    )

    print(
        independent_preview.columns.tolist()
    )

    if TARGET_COLUMN not in cv_preview.columns:

        raise ValueError(
            f"\nTARGET_COLUMN = '{TARGET_COLUMN}' "
            "was not found in the cross-validation dataset.\n\n"
            "Available columns are:\n"
            +
            "\n".join(
                f"  - {column}"
                for column in cv_preview.columns
            )
            +
            "\n\nSet TARGET_COLUMN at the top of this script "
            "to the actual hemolysis target column."
        )

    if TARGET_COLUMN not in independent_preview.columns:

        raise ValueError(
            f"\nTARGET_COLUMN = '{TARGET_COLUMN}' "
            "was not found in the independent dataset.\n\n"
            "Available columns are:\n"
            +
            "\n".join(
                f"  - {column}"
                for column in independent_preview.columns
            )
        )

    # ============================================================
    # Create pipeline
    # ============================================================

    pipeline = StackingRegressionPipeline(

        prott5_path=
            PROTT5_PATH,

        physio_path=
            PHYSIO_PATH,

        blosum_path=
            BLOSUM_PATH,

        sinu_path=
            SINU_PATH,

        random_state=
            RANDOM_STATE,
    )

    # ============================================================
    # Metric evaluator
    # ============================================================

    metric = Metric(
        "RegressionMetrics"
    )

    # ============================================================
    # Five-fold CV + independent test
    # ============================================================

    results = (
        pipeline.run_stacking_pipeline(

            cross_val_csv=
                CROSS_VAL_DATASET,

            independent_csv=
                INDEPENDENT_DATASET,

            metric_evaluator=
                metric,

            pooling="mean",
        )
    )

    # ============================================================
    # Save results
    # ============================================================

    pipeline.save_results(
        results,
        output_dir=
            OUTPUT_PATH,
        prefix=
            "PepMetaStack_hemolysis",
    )

    print("\n" + "=" * 70)
    print("HEMOLYSIS EXPERIMENT COMPLETED")
    print("=" * 70)