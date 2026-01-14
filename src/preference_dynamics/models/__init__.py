"""
Models module for preference dynamics parameter learning.

This module contains all parameter prediction models that implement the
PredictorModel protocol, including:
- CNN-based models (CNN1DPredictor)
- LSTM-based models (future)
- Physics-informed neural networks (PINN) (future)
- Feature-based models (future)

All models implement the PredictorModel protocol for unified training interface.
"""

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.models.cnn1d import CNN1DPredictor
from preference_dynamics.models.cnn1d_feat import CNN1DFeatPredictor
from preference_dynamics.models.cnn1d_residual import CNN1DResidualPredictor
from preference_dynamics.models.schemas import (
    CNN1DConfig,
    CNN1DFeatConfig,
    CNN1DResidualConfig,
    ModelConfig,
)
from preference_dynamics.models.utils import load_model, num_parameters, save_model

MODEL_REGISTRY: dict[str, tuple[type[ModelConfig], type[PredictorModel]]] = {
    "cnn1d": (CNN1DConfig, CNN1DPredictor),
    "cnn1d_feat": (CNN1DFeatConfig, CNN1DFeatPredictor),
    "cnn1d_residual": (CNN1DResidualConfig, CNN1DResidualPredictor),
}

__all__ = [
    "ModelConfig",
    "CNN1DConfig",
    "CNN1DFeatConfig",
    "CNN1DResidualConfig",
    "CNN1DPredictor",
    "CNN1DFeatPredictor",
    "CNN1DResidualPredictor",
    "PredictorModel",
    "MODEL_REGISTRY",
    "num_parameters",
    "load_model",
    "save_model",
]
