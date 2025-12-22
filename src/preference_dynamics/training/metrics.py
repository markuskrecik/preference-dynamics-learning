"""
Evaluation metrics for predictions.
"""

from typing import Literal

import numpy as np
import torch
from numpydantic import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preference_dynamics.schemas import (
    ICVector,
    ODEConfig,
    ODESolverConfig,
    ParameterVector,
    SolverConfig,
)
from preference_dynamics.solver.integrator import SolverConvergenceError, solve_ODE


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


def time_series_match_metrics(
    inferred_params: ParameterVector,
    inferred_ic: ICVector,
    observed_time_series: NDArray[Literal["*, *"], float],
    observed_time_points: NDArray[Literal["*"], float],
    solver_config: SolverConfig | None = None,
) -> dict[str, float | str]:
    """
    Evaluate time series match by solving ODE with inferred parameters/ICs.

    This function solves the ODE system using the inferred parameters and initial
    conditions, then compares the simulated time_series to the observed time_series.

    Args:
        inferred_params: Inferred parameter vector (θ̂)
        inferred_ic: Inferred initial conditions (x̂₀)
        observed_time_series: Observed time_series [u, a] of shape (2n, T)
        observed_time_points: Time points of shape (T,)
        solver_config: Solver configuration (default: uses observed time span)

    Returns:
        Dictionary with consistency metrics:
            - 'mse': Mean squared error between observed and simulated trajectories
            - 'mae': Mean absolute error
            - 'rmse': Root mean squared error
            - 'r2': R-squared score
            - 'max_error': Maximum absolute error across all time points
            - 'mean_relative_error': Mean relative error
    """
    n: int = inferred_params.n_actions  # type: ignore

    if observed_time_series.shape[0] != 2 * n:
        raise ValueError(
            f"Observed time_series must have shape (2n, T) for n={n}, "
            f"got {observed_time_series.shape}"
        )

    if observed_time_series.shape[1] != len(observed_time_points):
        raise ValueError(
            f"Time points length ({len(observed_time_points)}) must match "
            f"time_series time dimension ({observed_time_series.shape[1]})"
        )

    if solver_config is None:
        time_span = (float(observed_time_points[0]), float(observed_time_points[-1]))
        solver_config = SolverConfig(
            time_span=time_span,
            n_time_points=len(observed_time_points),
            solver_method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )

    ode_config = ODEConfig(parameters=inferred_params, initial_conditions=inferred_ic)
    config = ODESolverConfig(ode=ode_config, solver=solver_config)

    try:
        sample = solve_ODE(config)
        simulated_time_series = sample.time_series  # Shape: (2n, T)
    except SolverConvergenceError as e:
        return {
            "mse": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": float("-inf"),
            "max_error": float("inf"),
            "mre": float("inf"),
            "solver_success": False,
            "solver_message": str(e),
        }

    # Compute metrics
    observed_flat = observed_time_series.flatten()
    simulated_flat = simulated_time_series.flatten()

    mse = float(mean_squared_error(observed_flat, simulated_flat))
    mae = float(mean_absolute_error(observed_flat, simulated_flat))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(observed_flat, simulated_flat))

    errors = np.abs(observed_flat - simulated_flat)
    max_error = float(np.max(errors))

    epsilon = 1e-8
    relative_errors = errors / (np.abs(observed_flat) + epsilon)
    mre = float(np.mean(relative_errors))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_error": max_error,
        "mre": mre,
        "solver_success": True,
    }
