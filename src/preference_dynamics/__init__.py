"""
Preference Dynamics Parameter Learning

A machine learning research project for learning parameters of nonlinear
preference dynamics ODE systems from time series data.

This package supports variable dimensions (n actions) with primary focus on
n=3 (24 parameters, 6-dimensional time series).

Main components:
- solver: ODE solver and data generation
- data: Data pipeline, validation, and preprocessing
- models: ML model architectures (CNN, LSTM, PINN, feature-based)
- training: Training infrastructure and metrics
- experiments: Experiment orchestration and tracking
- visualization: Plotting utilities
"""

from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.experiments.runner import ExperimentRunner
from preference_dynamics.models import CNN1DConfig
from preference_dynamics.models.cnn1d import CNN1DPredictor
from preference_dynamics.schemas import RunnerConfig, TrainerConfig
from preference_dynamics.training.trainer import Trainer

__version__ = "0.1.0"
__author__ = "Markus Krecik"

VERSION = tuple(map(int, __version__.split(".")))


__all__ = [
    "__version__",
    "VERSION",
    "DataConfig",
    "DataManager",
    "CNN1DConfig",
    "CNN1DPredictor",
    "TrainerConfig",
    "Trainer",
    "RunnerConfig",
    "ExperimentRunner",
]
