import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)




from metrics import Metric

import data_retriever



# ==============================================================
# Feature Configuration
# ==============================================================

FEATURE_FLAGS = {
    "include_gmprot": True,
    "include_physio": True,
    "include_blosum": True,
    "include_sinu": True,
}


class StackingRegressionGMProtPipeline:

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self, alpha=10.0, feature_flags=None):

        self.alpha = alpha

        self.feature_flags = (
            feature_flags
            if feature_flags is not None
            else FEATURE_FLAGS.copy()
        )

    # ==========================================================
    # 1. Extract Features
    # ==========================================================

    @staticmethod
    def extract_features(record, feature_flags):
        """
        Extract sequence-level features from one record.

        GMProt:
            128-dimensional sequence-level embedding

        Physio:
            Mean-pooled if residue-level

        BLOSUM:
            Mean-pooled if residue-level

        Sinusoidal:
            Mean-pooled if residue-level
        """

        features = []

        # ------------------------------------------------------
        # GMProt
        # ------------------------------------------------------

        if feature_flags["include_gmprot"]:

            emb = np.asarray(
                record["emb"],
                dtype=np.float32,
            ).reshape(-1)

            features.append(emb)

        # ------------------------------------------------------
        # Physicochemical Features
        # ------------------------------------------------------

        if feature_flags["include_physio"]:

            physio = np.asarray(
                record["physio"],
                dtype=np.float32,
            )

            if physio.ndim > 1:
                physio = physio.mean(axis=0)

            features.append(
                physio.reshape(-1)
            )

        # ------------------------------------------------------
        # BLOSUM Features
        # ------------------------------------------------------

        if feature_flags["include_blosum"]:

            blosum = np.asarray(
                record["blosum"],
                dtype=np.float32,
            )

            if blosum.ndim > 1:
                blosum = blosum.mean(axis=0)

            features.append(
                blosum.reshape(-1)
            )

        # ------------------------------------------------------
        # Sinusoidal Features
        # ------------------------------------------------------

        if feature_flags["include_sinu"]:

            sinu = np.asarray(
                record["sino"],
                dtype=np.float32,
            )

            if sinu.ndim > 1:
                sinu = sinu.mean(axis=0)

            features.append(
                sinu.reshape(-1)
            )

        if not features:
            raise ValueError(
                "At least one feature must be enabled."
            )

        X = np.hstack(features)

        return X, float(record["mic"])

    # ==========================================================
    # 2. Prepare Dataset
    # ==========================================================

    @classmethod
    def prepare_split(
        cls,
        dataset_split,
        feature_flags,
    ):
        """
        Convert a list of records into X and y.
        """

        X_list = []
        y_list = []

        for record in dataset_split:

            X, y = cls.extract_features(
                record,
                feature_flags,
            )

            X_list.append(X)
            y_list.append(y)

        return (
            np.asarray(X_list, dtype=np.float32),
            np.asarray(y_list, dtype=np.float32),
        )

    # ==========================================================
    # 3. Build Stacking Ensemble
    # ==========================================================

    def build_stacking_ensemble(self):

        base_estimators = [

            (
                "ridge",
                Ridge(
                    alpha=self.alpha,
                ),
            ),

            (
                "svr",
                SVR(
                    C=10.0,
                    epsilon=0.1,
                    kernel="rbf",
                ),
            ),

            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=16,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),

            (
                "hist_gb",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]

        meta_learner = RidgeCV(
            alphas=np.logspace(-3, 3, 10)
        )

        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
        )

    # ==========================================================
    # 4. Analyze Meta-Learner Weights
    # ==========================================================

    def analyze_stacking_weights(
        self,
        stacking_model,
        fold_idx=1,
    ):
        """
        Analyze the coefficients assigned by the Ridge
        meta-learner to the base estimators.
        """

        estimator_names = [
            name
            for name, _ in stacking_model.estimators
        ]

        coefficients = np.asarray(
            stacking_model.final_estimator_.coef_
        ).reshape(-1)

        df_weights = pd.DataFrame({
            "Base Estimator": estimator_names,
            "Coefficient Weight": coefficients,
        })

        df_weights["Absolute Contribution"] = (
            df_weights["Coefficient Weight"].abs()
        )

        total_weight = (
            df_weights["Absolute Contribution"].sum()
        )

        if total_weight > 0:
            df_weights["Normalized Contribution (%)"] = (
                df_weights["Absolute Contribution"]
                / total_weight
                * 100
            )
        else:
            df_weights["Normalized Contribution (%)"] = 0.0

        df_weights = df_weights.sort_values(
            "Absolute Contribution",
            ascending=False,
        )

        print(
            "\n=== META-LEARNER WEIGHT DISTRIBUTION ==="
        )

        print(
            df_weights.to_string(index=False)
        )

        print(
            f"\nMeta-Learner Intercept: "
            f"{float(stacking_model.final_estimator_.intercept_):.4f}"
        )

        # ------------------------------------------------------
        # Save Plot
        # ------------------------------------------------------

        os.makedirs(
            "./results",
            exist_ok=True,
        )

        plt.figure(figsize=(10, 5))

        bars = plt.barh(
            df_weights["Base Estimator"],
            df_weights["Coefficient Weight"],
        )

        plt.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        plt.xlabel(
            "Meta-Learner Coefficient"
        )

        plt.ylabel(
            "Base Estimator"
        )

        plt.title(
            f"Stacking Meta-Learner Weights - Fold {fold_idx}"
        )

        plt.gca().invert_yaxis()

        for bar in bars:

            width = bar.get_width()

            offset = (
                0.01
                if width >= 0
                else -0.03
            )

            plt.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
            )

        plt.tight_layout()

        output_path = (
            f"./results/"
            f"stacking_meta_learner_weights_"
            f"fold_{fold_idx}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nMeta-learner weight plot saved to: "
            f"{output_path}"
        )

        return df_weights

    # ==========================================================
    # 5. Run Fold 1
    # ==========================================================

    def run_fold(
        self,
        data_splits,
        metric_evaluator=None,
        fold_index=1,
    ):
        """
        Train and evaluate the stacking model on one CV fold.

        Fold 1:
            train -> data_splits["train"]
            val   -> data_splits["val"]
            test  -> data_splits["test"]
        """

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Running CV Fold {fold_index}"
        )

        print(
            f"{'=' * 60}"
        )

        # ======================================================
        # Prepare Data
        # ======================================================

        X_train, y_train = self.prepare_split(
            data_splits["train"],
            self.feature_flags,
        )

        X_val, y_val = self.prepare_split(
            data_splits["val"],
            self.feature_flags,
        )

        X_test, y_test = self.prepare_split(
            data_splits["test"],
            self.feature_flags,
        )

        print(
            f"\nTrain: {X_train.shape}"
        )

        print(
            f"Validation: {X_val.shape}"
        )

        print(
            f"Test: {X_test.shape}"
        )

        print(
            f"Feature dimension: {X_train.shape[1]}"
        )

        # ======================================================
        # Standardization
        # ======================================================

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_val = scaler.transform(
            X_val
        )

        X_test = scaler.transform(
            X_test
        )

        # ======================================================
        # Build Model
        # ======================================================

        model = self.build_stacking_ensemble()

        # ======================================================
        # Train
        # ======================================================

        print(
            "\nTraining stacking model..."
        )

        model.fit(
            X_train,
            y_train,
        )

        # ======================================================
        # Validation Prediction
        # ======================================================

        val_preds = model.predict(
            X_val
        )

        # ======================================================
        # Test Prediction
        # ======================================================

        test_preds = model.predict(
            X_test
        )

        # ======================================================
        # Metrics
        # ======================================================

        if metric_evaluator is not None:

            val_metrics = (
                metric_evaluator.compute_metrics(
                    y_val,
                    val_preds,
                )
            )

            test_metrics = (
                metric_evaluator.compute_metrics(
                    y_test,
                    test_preds,
                )
            )

            print(
                "\n=== VALIDATION RESULTS ==="
            )

            for key, value in val_metrics.items():
                print(
                    f"{key:<15}: {value:.4f}"
                )

            print(
                "\n=== TEST RESULTS ==="
            )

            for key, value in test_metrics.items():
                print(
                    f"{key:<15}: {value:.4f}"
                )

        # ======================================================
        # Meta-Learner Analysis
        # ======================================================

        self.analyze_stacking_weights(
            model,
            fold_idx=fold_index,
        )

        return {
            "model": model,
            "scaler": scaler,
            "val_metrics": val_metrics if metric_evaluator else None,
            "test_metrics": test_metrics if metric_evaluator else None,
            "val_predictions": val_preds,
            "test_predictions": test_preds,
        }


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":

    # ==========================================================
    # Dataset / Feature Files
    # ==========================================================

    data_splits = data_retriever.load_features(
        fused_embedding_file=(
            "./data/gmprot_ecoli_fusion_embeddings.npy"
        ),
        blosum_csv=(
            "./data/ecoli_blosum62_features.csv"
        ),
        physio_csv=(
            "./data/ecoli_phyiochem.csv"
        ),
        sinusoidal_csv=(
            "./data/ecoli_sinusoidal_encoding.csv"
        ),
        dataset_csv=(
            "./data/ecoli_five_fold_dataset.csv"
        ),
        fold_index=3,
    )

    # ==========================================================
    # Metric
    # ==========================================================

    metric = Metric(
        "RegressionMetrics"
    )

    # ==========================================================
    # Pipeline
    # ==========================================================

    pipeline = StackingRegressionGMProtPipeline(
        alpha=10.0,
        feature_flags=FEATURE_FLAGS,
    )

    # ==========================================================
    # Run Fold 1
    # ==========================================================

    results = pipeline.run_fold(
        data_splits=data_splits,
        metric_evaluator=metric,
        fold_index=1,
    )