"""
Model Adapters for flexible model inputs and targets specification from arbitrary data sources.
Ingested by torch.utils.data.Dataset.
"""

from typing import Any, Protocol

import torch

from preference_dynamics.schemas import TimeSeriesSample


class ModelAdapter(Protocol):
    """
    Adapter for model input and output.
    """

    def __call__(self, sample: Any) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]: ...

    def n_inputs(self, sample: Any) -> int: ...

    def n_outputs(self, sample: Any) -> int: ...


class CNN1DParamAdapter:
    """
    Adapter for CNN1D parameter prediction.
    Compatible with `model.forward(x)`.
    """

    def __call__(
        self, sample: TimeSeriesSample
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        x = torch.from_numpy(sample.time_series.copy()).float()
        target = torch.from_numpy(sample.parameters.values.copy()).float()

        return {"inputs": {"x": x}, "target": target}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(sample.time_series.shape[0])

    def n_outputs(self, sample: TimeSeriesSample) -> int:
        return int(sample.parameters.values.shape[0])


class CNN1DParamICAdapter:
    """
    Adapter for CNN1D parameter and initial conditions prediction.
    Compatible with `model.forward(x)`.
    """

    def __call__(
        self, sample: TimeSeriesSample
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        x = torch.from_numpy(sample.time_series.copy()).float()
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        target = torch.cat([parameters, initial_conditions], dim=0)

        return {"inputs": {"x": x}, "target": target}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(sample.time_series.shape[0])

    def n_outputs(self, sample: TimeSeriesSample) -> int:
        return int(sample.parameters.values.shape[0] + sample.initial_conditions.values.shape[0])
