"""
Experiment infrastructure for preference dynamics parameter learning.

This module provides:
- ExperimentRunner: Orchestrates training with different configs and MLflow tracking
"""

from preference_dynamics.experiments.experiment import Experiment
from preference_dynamics.experiments.runner import (
    ExperimentRunner,
)

__all__ = ["Experiment", "ExperimentRunner"]
