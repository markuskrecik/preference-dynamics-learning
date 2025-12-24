"""
Logging utilities.
"""

import mlflow

from preference_dynamics.schemas import InversePINNEvaluationReport
from preference_dynamics.utils import if_logging


@if_logging
def log_objective_terms(
    step: int,
    data_loss: float | None = None,
    physics_loss: float | None = None,
    supervised_loss: float | None = None,
) -> None:
    """
    Log all objective term losses to MLflow in a single call.

    Args:
        step: Training step (epoch or batch)
        data_loss: Data fitting loss (optional)
        physics_loss: Physics residual loss (optional)
        supervised_loss: Supervised parameter/IC loss (optional)
    """
    if data_loss is not None:
        mlflow.log_metric("loss_data", data_loss, step=step)
    if physics_loss is not None:
        mlflow.log_metric("loss_physics", physics_loss, step=step)
    if supervised_loss is not None:
        mlflow.log_metric("loss_supervised", supervised_loss, step=step)


@if_logging
def log_evaluation_report(
    report: InversePINNEvaluationReport,
    split: str = "test",
    step: int | None = None,
) -> None:
    """
    Log inverse PINN evaluation report metrics to MLflow.
        - parameter_error (median relative error of inferred parameters)
        - ic_error (median relative error of inferred initial conditions)
        - trajectory_error (trajectory reconstruction error)
        - physics_residual_score (physics residual on reconstructed trajectories)

    Args:
        report: Evaluation report with metrics
        split: Dataset split name (train/val/test)
        step: Optional step number for metric logging
    """
    prefix = f"eval_{split}_"

    mlflow.log_metric(f"{prefix}trajectory_error", report.trajectory_error, step=step)
    mlflow.log_metric(f"{prefix}physics_residual_score", report.physics_residual_score, step=step)

    if report.parameter_error is not None:
        mlflow.log_metric(f"{prefix}parameter_error", report.parameter_error, step=step)
    if report.ic_error is not None:
        mlflow.log_metric(f"{prefix}ic_error", report.ic_error, step=step)

    for key, value in report.time_series_match.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{prefix}time_series_match_{key}", value, step=step)
