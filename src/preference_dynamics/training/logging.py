"""
Logging utilities.
"""

import mlflow

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
