import pickle
import numpy as np

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)



from recomb_code.data_retriever import DataRetriever
from metrics import Metric

metric = Metric("RegressionMetrics")

feature_flags = {
        "include_emb": True,
        "include_physio": True,
        "include_blosum": True,
        "include_sinu": True,
    }

class StackingRegressionSVDGuidedPipeline:
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

        def svd_attentive_pool(arr):
            """
            Extracts the 1st Principal Component (dominant axis of variance) across residues.
            arr shape: (seq_len, 1024)
            Returns: (1024,) 1D vector
            """
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                return arr

            # SVD: arr = U @ S @ Vt
            # Vt[0] is the 1st right singular vector (1024-dim)
            _, S, Vt = np.linalg.svd(arr, full_matrices=False)
            
            # Scale by top singular value to retain magnitude
            return Vt[0] * S[0]

        features = []

        # Selectively pool and append enabled feature modalities
        if include_emb:
            features.append(svd_attentive_pool(emb)) 
            
        if include_physio:
            features.append(svd_attentive_pool(physio))
            
        if include_blosum:
            features.append(svd_attentive_pool(blosum))
            
        if include_sinu:
            features.append(svd_attentive_pool(sinu))

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
        base_estimators = [
            ("ridge", Ridge(alpha=self.alpha)),
            ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
            ("extra_trees", ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42)),
            ("hist_gb", HistGradientBoostingRegressor(max_iter=100, l2_regularization=1.0, random_state=42)),
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


if __name__ == "__main__":
    DATASET_PATH = "./ecoli_prott5_based_datasets.pkl"
    
    # Initialize Data Retriever (assumed class defined elsewhere)
    data_retriever = DataRetriever(DATASET_PATH)
    data_retriever.explain_data_structure()
    
    # Instantiate pipeline and execute
    pipeline = StackingRegressionSVDGuidedPipeline(alpha=10.0)
    pipeline.run_stacking_pipeline(data_retriever, metric_evaluator=metric)