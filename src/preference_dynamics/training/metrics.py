"""
Evaluation metrics for predictions.
"""

from typing import Literal

import numpy as np
import torch
from numpydantic import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: NDArray[Literal["*, *"], float],
    y_pred: NDArray[Literal["*, *"], float],
) -> dict[str, float | NDArray[Literal["*"], float]]:
    """
    Computes aggregate metrics (MAE, RMSE, R², MAPE) and per-target metrics.

    Args:
        y_true: True parameter values
        y_pred: Predicted parameter values

    Returns:
        metrics: Dictionary with keys:
            - 'mse': Mean squared error (scalar)
            - 'mae': Mean absolute error (scalar)
            - 'rmse': Root mean squared error (scalar)
            - 'r2': R-squared score (scalar)
            - 'mre': Mean relative error (scalar)
            - 'nrmse_range': Normalized RMSE by target range (scalar)
            - 'nrmse_std': Normalized RMSE by target std (scalar)
            - 'pearson_r': Pearson correlation coefficient (scalar)
            - 'spearman_r': Spearman rank correlation (scalar)
            - 'medae': Median absolute error (scalar)
            - 'per_target_mae': Array with MAE per parameter
            - 'per_target_mse': Array with MSE per parameter
            - 'per_target_mre': Array with mean relative error per parameter
            - 'per_target_nrmse': Array with normalized RMSE per parameter
    """

    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, got {y_true.shape} and {y_pred.shape}"
        )

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    epsilon = 1e-8
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    relative_errors = abs_errors / (np.abs(y_true) + epsilon)
    mre = float(np.mean(relative_errors))

    target_range = float(np.max(y_true) - np.min(y_true))
    target_std = float(np.std(y_true))
    nrmse_range = float(rmse / (target_range + epsilon))
    nrmse_std = float(rmse / (target_std + epsilon))

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    pearson_r = float(pearsonr(y_true_flat, y_pred_flat)[0])
    spearman_r = float(spearmanr(y_true_flat, y_pred_flat)[0])

    medae = float(np.median(abs_errors))

    per_target_mae = np.mean(abs_errors, axis=0)
    per_target_mse = np.mean(errors**2, axis=0)

    per_target_mre = np.mean(relative_errors, axis=0)
    per_target_rmse = np.sqrt(per_target_mse)
    per_target_range = np.max(y_true, axis=0) - np.min(y_true, axis=0)
    per_target_nrmse = per_target_rmse / (per_target_range + epsilon)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        # "mape": mape,
        "mre": mre,
        "nrmse_range": nrmse_range,
        "nrmse_std": nrmse_std,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "medae": medae,
        "per_target_rmse": per_target_rmse,
        "per_target_mae": per_target_mae,
        "per_target_mre": per_target_mre,
        "per_target_nrmse": per_target_nrmse,
    }
