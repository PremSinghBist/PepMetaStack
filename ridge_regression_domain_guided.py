import os
import pickle
import numpy as np

import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.base import clone
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)



from dataset import DataRetriever
from metrics import Metric
from physico_util import get_domain_attention



feature_flags = {
        "include_emb": True,
        "include_physio": True,
        "include_blosum": True,
        "include_sinu": True,
    }

class StackingRegressionDomainGuidedPipeline:
    def __init__(self, alpha=10.0, feature_flags=None):
        self.alpha = alpha
        
        # Default feature flags if none provided
        if feature_flags is None:
            self.feature_flags = {
                "include_emb": True,
                "include_physio": True,
                "include_blosum": True,
                "include_sinu": True,
            }
        else:
            self.feature_flags = feature_flags

    # ==========================================
    # 1. Feature Extraction & Pooling Function
    # ==========================================
    @staticmethod
    def extract_features(
        record,
        include_emb=True,
        include_physio=True,
        include_blosum=True,
        include_sinu=True,
    ):
        """
        Extracts and pools multi-modal features (ProtT5, Physio, BLOSUM, Sinu) 
        into a single 1D concatenated feature vector based on selected feature flags.
        
        Returns:
            tuple: (X_vector, mic) where X_vector is a 1D np.ndarray and mic is a float.
        """
        emb, physio, blosum, sinu, seq, mic = record

        def domain_guided_pool(arr):
            """Pools 2D residue-level arrays to 1D sequence representations.
               Domain-guided attention is applied to emphasize functionally relevant regions.
               The attention is based on hydrophobicity and charge of amino acids, computed via get_domain_attention.
            """
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                
                attention = get_domain_attention(seq[:arr.shape[0]])  # Ensure attention matches the number of residues in arr
                
                return np.sum(arr * attention[:, np.newaxis], axis=0) #Broadcast attn to match feature dimension | then sum across residues
            return arr #Vector of shape (feature_dim,) 
        
        

        features = []

        # Selectively pool and append enabled feature modalities
        if include_emb:
            features.append(domain_guided_pool(emb)) 
            
        if include_physio:
            features.append(domain_guided_pool(physio))
            
        if include_blosum:
            features.append(domain_guided_pool(blosum))
            
        if include_sinu:
            features.append(domain_guided_pool(sinu))

        # Guard against empty feature selection
        if not features:
            raise ValueError("At least one feature flag must be set to True.")

        # Concatenate all selected pooled feature vectors horizontally
        X_vector = np.hstack(features)
        
        return X_vector, float(mic)

    # ==========================================
    # 2. Data Split Preparation
    # ==========================================
    @classmethod
    def prepare_split(cls, dataset_split, **feature_flags):
        """
        Converts a fold split (list of records) into design matrix X and target y.
        """
        X_list, y_list = [], []
        for record in dataset_split:
            X_vec, y_val = cls.extract_features(record, **feature_flags)
            X_list.append(X_vec)
            y_list.append(y_val)
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    # ==========================================
    # 3. Model Definition: Stacking Regressor
    # ==========================================
    def build_stacking_ensemble(self):
        """
        Constructs a diverse Stacking Ensemble Regressor.
        """
        '''base_estimators = [
            ("ridge", Ridge(alpha=self.alpha)),
            ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
            ("extra_trees", ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42)),
            ("hist_gb", HistGradientBoostingRegressor(max_iter=100, l2_regularization=1.0, random_state=42)),
        ]'''
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from xgboost import XGBRegressor
        base_estimators = [
            # --- Existing Models ---
            ("ridge", Ridge(alpha=10.0)),
            ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=150, max_depth=16, random_state=42, n_jobs=-1
                ),
            ),
            (
                "hist_gb",
                HistGradientBoostingRegressor(
                    max_iter=150, l2_regularization=1.0, random_state=42
                ),
            ),
            # --- New Complementary Models ---
            # 1. Local Manifold / Sequence Similarity Search
            (
                "knn_cosine",
                KNeighborsRegressor(
                    n_neighbors=7, weights="distance", metric="cosine", n_jobs=-1
                ),
            ),
            # 2. Non-linear Dense Feature Interaction
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=0.01,
                    max_iter=300,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
            # 3. Regularized Gradient Boosting with Subsampling
            (
                "xgb",
                XGBRegressor(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            # 4. Latent Component Projection for Collinear Embeddings
            ("pls", PLSRegression(n_components=15)),
        ]
        
        meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 10))

        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,  # Internal 5-fold CV for meta-feature generation
            n_jobs=-1,
        )
        return stacking_model

    @staticmethod
    def build_tuned_stacking_ensemble(X_train, y_train, random_state=42):
        
        """
        1. Runs inner 3-fold CV on X_train to find optimal hyperparameters for each base model.
        2. Constructs a StackingRegressor using the best-tuned base models.
        """
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        best_params_summary = {}
        
        print("  [Tuning Base Estimators via Inner CV...]")

        # -------------------------------------------------------------
        # 1. Tune Ridge Regression (Regularization Strength)
        # -------------------------------------------------------------
        ridge_grid = {
            'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]  # Cap alpha to prevent underfitting
        }
        ridge_search = GridSearchCV(
            Ridge(), ridge_grid, cv=inner_cv, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        ridge_search.fit(X_train, y_train)
        best_ridge = ridge_search.best_estimator_
        best_params_summary["ridge"] = ridge_search.best_params_

        # -------------------------------------------------------------
        # 2. Tune SVR (C, Gamma, Epsilon)
        # Crucial for 1024-dim ProtT5 vectors where Gamma heavily affects RBF kernel width
        # -------------------------------------------------------------
        svr_grid = {
            'C': [10.0, 50.0, 100.0],                # Higher capacity
            'gamma': ['scale', 0.01],               # Keep kernel expressive
            'epsilon': [0.05, 0.1]
        }
        
        svr_search = GridSearchCV(
            SVR(kernel="rbf"), svr_grid, cv=inner_cv, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        svr_search.fit(X_train, y_train)
        best_svr = svr_search.best_estimator_
        best_params_summary['svr'] = svr_search.best_params_

        # -------------------------------------------------------------
        # 3. Tune Extra Trees Regressor (Tree depth & split constraints)
        # -------------------------------------------------------------
        et_grid = {
            'n_estimators': [100, 200],
            'max_depth': [12, 20, None],            # Allow full tree depth
            'min_samples_split': [2]
        }
        
        et_search = GridSearchCV(
            ExtraTreesRegressor(random_state=random_state),
            et_grid, cv=inner_cv, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        et_search.fit(X_train, y_train)
        best_et = et_search.best_estimator_
        best_params_summary['extra_trees'] = et_search.best_params_

        # -------------------------------------------------------------
        # 4. Tune HistGradientBoosting (Learning rate & Regularization)
        # -------------------------------------------------------------
        hgb_grid = {
            'learning_rate': [0.03, 0.05, 0.1],
            'l2_regularization': [0.1, 1.0, 10.0],
            'max_iter': [100, 150]
        }
        hgb_search = GridSearchCV(
            HistGradientBoostingRegressor(random_state=random_state),
            hgb_grid, cv=inner_cv, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        hgb_search.fit(X_train, y_train)
        best_hgb = hgb_search.best_estimator_
        best_params_summary['hist_gb'] = hgb_search.best_params_
        
        # -------------------------------------------------------------
        # 5. Assemble Stacking Ensemble with Tuned Models
        # -------------------------------------------------------------
        base_estimators = [
            ("ridge", best_ridge),
            ("svr", best_svr),
            ("extra_trees", best_et),
            ("hist_gb", best_hgb),
        ]
        
        

        # Meta-learner: RidgeCV evaluates multiple alphas automatically during stacking fit
        meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 15))

        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,  # Outer 5-fold CV to generate meta-features
            n_jobs=-1,
        )

        return stacking_model, best_params_summary
    
    ## Ablation studies 
    def get_base_estimator_dict(self , alpha=10.0, random_state=42):
        """Returns a dictionary of un-fitted candidate base estimators."""
        return {
            "ridge": Ridge(alpha=alpha),
            "svr": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
            "extra_trees": ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=random_state),
            "hist_gb": HistGradientBoostingRegressor(max_iter=100, l2_regularization=1.0, random_state=random_state),
    }
    
    def run_model_ablation(self, folds, feature_flags, pipeline_obj, metric_evaluator):
        """
        Evaluates individual models and all stacking ensemble combinations across all 5 folds.
        """
        candidate_estimators = self.get_base_estimator_dict(alpha=pipeline_obj.alpha)
        model_names = list(candidate_estimators.keys())

        # Build all unique combinations (1-model standalone, 2-model, 3-model, 4-model)
        all_combinations = []
        for r in range(1, len(model_names) + 1):
            for combo in itertools.combinations(model_names, r):
                all_combinations.append(combo)

        # Store metrics across folds: { combo_key: list_of_fold_metric_dicts }
        results = { "+".join(combo): [] for combo in all_combinations }

        # Outer 5-Fold Loop
        for fold_idx, (train_set, val_set, test_set) in enumerate(folds):
            print(f"--- Processing Fold {fold_idx + 1}/{len(folds)} for Ablation ---")

            # Prepare and scale features strictly on train set
            X_train, y_train = pipeline_obj.prepare_split(train_set, **feature_flags)
            X_test, y_test = pipeline_obj.prepare_split(test_set, **feature_flags)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Iterate through every subset combination
            for combo in all_combinations:
                combo_key = "+".join(combo)

                # Single Standalone Model (No Stacking meta-learner needed)
                if len(combo) == 1:
                    model = clone(candidate_estimators[combo[0]])
                    model.fit(X_train_scaled, y_train)
                    test_preds = model.predict(X_test_scaled)

                # Stacking Ensemble (2+ base estimators)
                else:
                    estimators_list = [(name, clone(candidate_estimators[name])) for name in combo]
                    meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 10))
                    model = StackingRegressor(
                        estimators=estimators_list,
                        final_estimator=meta_learner,
                        cv=5,
                        n_jobs=-1,
                    )
                    model.fit(X_train_scaled, y_train)
                    test_preds = model.predict(X_test_scaled)

                # Compute and record test set metrics
                metrics = metric_evaluator.compute_metrics(y_test, test_preds)
                results[combo_key].append(metrics)

        # Aggregate Results Across Folds into a Summary DataFrame
        summary_rows = []
        for combo_key, metrics_list in results.items():
            row = {
                "Configuration": combo_key,
                "Type": "Standalone" if "+" not in combo_key else f"Stacking ({combo_key.count('+') + 1} models)",
                "Pearson_R": f"{np.mean([m['Pearson_R'] for m in metrics_list]):.4f} ± {np.std([m['Pearson_R'] for m in metrics_list]):.4f}",
                "Spearman_R": f"{np.mean([m['Spearman_R'] for m in metrics_list]):.4f} ± {np.std([m['Spearman_R'] for m in metrics_list]):.4f}",
                "RMSE": f"{np.mean([m['RMSE'] for m in metrics_list]):.4f} ± {np.std([m['RMSE'] for m in metrics_list]):.4f}",
                "R2": f"{np.mean([m['R2'] for m in metrics_list]):.4f} ± {np.std([m['R2'] for m in metrics_list]):.4f}",
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def save_ablation_results(self,
        df: pd.DataFrame,
        output_dir: str = "./results",
        filename_prefix: str = "model_ablation_summary"
    ) -> str:
        """Saves the ablation summary DataFrame to a CSV file.

        Returns:
        --------
        str: Path to the saved CSV file.
        """
        
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{filename_prefix}.csv"

        filepath = os.path.join(output_dir, filename)

        # Export to CSV without pandas row index
        df.to_csv(filepath, index=False)
        print(f"Ablation summary successfully saved to: {filepath}")

        return filepath
    
    # ==========================================
    # 4. Main Training and Evaluation Loop
    # ==========================================
    def run_stacking_pipeline(self, data_retriever, metric_evaluator=None):
        """Loads 5-fold dataset, trains Stacking Regressor across folds, and logs results."""
        folds = data_retriever.load_dataset_pickle()
        print(f"Successfully loaded {len(folds)} folds.\n")

        val_metrics_list = []
        test_metrics_list = []
        # all_fold_params = []  # To store best hyperparameters for each fold

        # Running training for each fold
        for fold_idx, (train_set, val_set, test_set) in enumerate(folds):
            print(f"--- Training Fold {fold_idx + 1}/{len(folds)} ---")

            # 1. Prepare Feature Matrices
            X_train, y_train = self.prepare_split(train_set, **self.feature_flags)
            X_val, y_val = self.prepare_split(val_set, **self.feature_flags)
            X_test, y_test = self.prepare_split(test_set, **self.feature_flags)

            # 2. Scale Features (Fit strictly on Train to avoid data leakage)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # 3. Train Stacking Model
            model = self.build_stacking_ensemble()
            # model, fold_params = self.build_tuned_stacking_ensemble(X_train_scaled, y_train) #Hyperparameter tuning for each base model
            # all_fold_params.append(fold_params)
            
            # 3. Log tuned parameters for current fold
            '''print(f"--- Best Hyperparameters (Fold {fold_idx + 1}) ---")
            for model_name, params in fold_params.items():
                formatted_params = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"  ├── {model_name:<12}: {formatted_params}")'''
                
            model.fit(X_train_scaled, y_train)
            
            # --- ADD THIS LINE TO CALL IT FOR EVERY FOLD ---
            print(f"\n--- Meta-Learner Weights Analysis (Fold {fold_idx + 1}) ---")
            self.analyze_stacking_weights(model, fold_idx+1)

            # 4. Predict
            val_preds = model.predict(X_val_scaled)
            test_preds = model.predict(X_test_scaled)

            # 5. Evaluate Metrics
            if metric_evaluator:
                val_m = metric_evaluator.compute_metrics(y_val, val_preds)
                test_m = metric_evaluator.compute_metrics(y_test, test_preds)

                val_metrics_list.append(val_m)
                test_metrics_list.append(test_m)

                print(
                    f"Fold {fold_idx + 1} Test Results -> "
                    f"Pearson R: {test_m['Pearson_R']:.4f} | "
                    f"Spearman R: {test_m['Spearman_R']:.4f} | "
                    f"Kendall Tau: {test_m['Kendall_Tau']:.4f} | "
                    f"RMSE: {test_m['RMSE']:.4f} | "
                    f"R2: {test_m['R2']:.4f}"
                )

        # ==========================================
        # 5. Summary Across All Folds
        # ==========================================
        if test_metrics_list:
            print("\n" + "=" * 65)
            print("=== FINAL 5-FOLD CROSS-VALIDATION SUMMARY (TEST SET) ===")
            print("=" * 65)

            metrics_keys = ["Pearson_R", "Spearman_R", "Kendall_Tau", "RMSE", "MAE", "R2"]
            
            for key in metrics_keys:
                if key in test_metrics_list[0]:
                    values = [m[key] for m in test_metrics_list]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  ├── {key:<12}: {mean_val:.4f} ± {std_val:.4f}")

            print("=" * 65)
            
        print("\nFeatures included in the model:")
        for feature, enabled in self.feature_flags.items():
            if enabled:
                print(f"  ├── {feature}")
                
    
    def analyze_stacking_weights(self, stacking_model, fold_idx: int =0):
        """Extracts and visualizes the feature weights assigned by the final meta-learner

        to each base estimator inside a fitted StackingRegressor.
        """
        # 1. Retrieve estimator names and corresponding meta-learner coefficients
        estimator_names = [name for name, _ in stacking_model.estimators]
        meta_learner = stacking_model.final_estimator_
        coefficients = meta_learner.coef_

        # 2. Construct summary DataFrame
        df_weights = pd.DataFrame(
            {
                "Base Estimator": estimator_names,
                "Coefficient Weight": coefficients,
                "Absolute Contribution": np.abs(coefficients),
            }
        ).sort_values(by="Absolute Contribution", ascending=False)

        # Compute percentage contribution relative to total absolute weight
        total_abs_weight = df_weights["Absolute Contribution"].sum()
        df_weights["Normalized Contribution (%)"] = (
            df_weights["Absolute Contribution"] / total_abs_weight
        ) * 100

        print("\n=== META-LEARNER WEIGHT DISTRIBUTION ===")
        print(df_weights.to_string(index=False))
        print(f"\nMeta-Learner Intercept: {meta_learner.intercept_:.4f}")

        # 3. Plot Horizontal Bar Chart
        plt.figure(figsize=(10, 5))
        palette = sns.color_palette("mako", len(df_weights))

        bars = plt.barh(
            df_weights["Base Estimator"],
            df_weights["Coefficient Weight"],
            color=palette,
            edgecolor="black",
        )

        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Meta-Learner Coefficient (Ridge Weight)", fontsize=11)
        plt.ylabel("Base Estimator", fontsize=11)
        plt.title("Stacking Meta-Learner Weight Attribution", fontsize=13, pad=12)
        plt.gca().invert_yaxis()  # Highest weight at the top
        plt.grid(axis="x", linestyle=":", alpha=0.6)

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            offset = 0.01 if width >= 0 else -0.03
            plt.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(f"results/stacking_meta_learner_weights_fold_{fold_idx}.png", dpi=300, bbox_inches="tight")
        print(f"\nMeta-learner weight distribution plot saved as 'results/stacking_meta_learner_weights_fold_{fold_idx}.png'.")
        plt.show()

        return df_weights


if __name__ == "__main__":
    DATASET_PATH = "./ecoli_prott5_based_datasets.pkl"
    
    # Initialize Data Retriever (assumed class defined elsewhere)
    data_retriever = DataRetriever(DATASET_PATH)
    data_retriever.explain_data_structure()
    
    # 1. Load folds
    folds = data_retriever.load_dataset_pickle()
    metric = Metric("RegressionMetrics")
    
    # Instantiate pipeline and execute
    pipeline = StackingRegressionDomainGuidedPipeline(alpha=10.0)
    pipeline.run_stacking_pipeline(data_retriever, metric_evaluator=metric)
    
    ## 2. Run model ablation study
    '''print("\nStarting Model Ablation Study...")
    ablation_df = pipeline.run_model_ablation(
        folds=folds,
        feature_flags=pipeline.feature_flags,
        pipeline_obj=pipeline,
        metric_evaluator=metric,
    )
    pipeline.save_ablation_results(ablation_df, output_dir="./results", filename_prefix="model_ablation_summary")'''
    
    