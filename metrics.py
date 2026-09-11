import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
class Metric:
    def __init__(self, name):
        self.name = name

    def compute_metrics(self, y_true, y_pred):
        """Computes regression evaluation metrics: Pearson R, Spearman Rho, Kendall Tau, RMSE, MAE, R2."""
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_r, _ = stats.spearmanr(y_true, y_pred)
        kendall_tau, _ = stats.kendalltau(y_true, y_pred)  
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "Pearson_R": pearson_r,
            "Spearman_R": spearman_r,
            "Kendall_Tau": kendall_tau,  # <-- Included in returned dictionary
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }