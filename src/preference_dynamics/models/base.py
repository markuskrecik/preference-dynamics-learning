"""
Base model interface and utilities for parameter prediction models.
"""

from abc import ABC, abstractmethod

from torch import nn

from preference_dynamics.models.schemas import ModelConfig


class PredictorModel(nn.Module, ABC):  # type: ignore
    """
    Base class for prediction models.

    All models that predict parameters from time series should inherit from this class.
    """

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        """
        Configuration for the predictor model.
        """
        ...

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Return number of trainable parameters in model."""
        ...
