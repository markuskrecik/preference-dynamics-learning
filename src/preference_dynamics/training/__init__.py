"""
Training module for preference dynamics parameter learning.

This module provides:
- Trainer: Unified training class for neural network models
- TrainingHistory: Training results and metrics tracking
- compute_metrics: Evaluation metrics computation
- MLflow integration for experiment tracking
"""

from preference_dynamics.training.metrics import compute_metrics
from preference_dynamics.training.trainer import Trainer

__all__ = [
    "Trainer",
    "compute_metrics",
]
