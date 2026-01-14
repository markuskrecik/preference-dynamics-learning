"""
Logging utilities.
"""

import mlflow
import torch

from preference_dynamics.utils import if_logging

@if_logging
def log_PINN_loss(loss: dict[str, float | torch.Tensor], step: int) -> None:
    """
    Log PINN loss terms to MLflow.

    Args:
        loss: Dictionary with "data", "physics", "supervised", "total" keys
        step: Training step (epoch or batch)
    """
    if "data" in loss:
        data_val = loss["data"]
        if isinstance(data_val, torch.Tensor):
            data_val = data_val.item()
        mlflow.log_metric("loss_data", data_val, step=step)
    if "physics" in loss:
        physics_val = loss["physics"]
        if isinstance(physics_val, torch.Tensor):
            physics_val = physics_val.item()
        mlflow.log_metric("loss_physics", physics_val, step=step)
    if "supervised" in loss:
        supervised_val = loss["supervised"]
        if isinstance(supervised_val, torch.Tensor):
            supervised_val = supervised_val.item()
        mlflow.log_metric("loss_supervised", supervised_val, step=step)
    if "total" in loss:
        total_val = loss["total"]
        if isinstance(total_val, torch.Tensor):
            total_val = total_val.item()
        mlflow.log_metric("loss_total", total_val, step=step)

