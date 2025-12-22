"""
Training module for preference dynamics parameter learning.

This module provides:
- Trainer: Unified training class for neural network models
- TrainingHistory: Training results and metrics tracking
- compute_metrics: Evaluation metrics computation
- MLflow integration for experiment tracking
- Logging utilities for inverse PINN training
- Loss computation utilities for physics-informed training
- Simulation consistency evaluation utilities
"""

from preference_dynamics.training.logging import (
    log_objective_terms,
)
from preference_dynamics.training.loss import (
    compute_physics_residual_score,
)
from preference_dynamics.training.metrics import (
    compute_metrics,
    time_series_match_metrics,
)
from preference_dynamics.training.trainer import Trainer

__all__ = [
    "Trainer",
    "compute_metrics",
    "time_series_match_metrics",
    "log_objective_terms",
    "compute_physics_residual_score",
]
