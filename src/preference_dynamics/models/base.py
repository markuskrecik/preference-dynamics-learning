"""
Base model interface and utilities for parameter prediction models.
"""

from abc import ABC

import torch

from preference_dynamics.models.schemas import ModelConfig


class PredictorModel(torch.nn.Module, ABC):  # type: ignore
    """
    Base class for prediction models.

    All models that predict parameters from time series should inherit from this class.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

    @property
    @torch.jit.unused  # type: ignore
    def config(self) -> ModelConfig:
        """
        Configuration for the predictor model.
        """
        return self._config
