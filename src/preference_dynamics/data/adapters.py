"""
Model Adapters for flexible model inputs and targets specification from arbitrary data sources.
Ingested by torch.utils.data.Dataset.
"""

from typing import Any, Protocol

import numpy as np
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


class CNN1DParamICForecastAdapter:
    """
    Adapter for CNN1D parameter and initial conditions and forecast values prediction.
    Compatible with `model.forward(x)`.
    """

    def inputs(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(sample.time_series.copy()).float()
        return {"x": x}

    def target(self, sample: TimeSeriesSample) -> torch.Tensor:
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        forecast_values = torch.from_numpy(np.array(sample.features["forecast_values"])).float()
        forecast_values = forecast_values.flatten()
        target = torch.cat([parameters, initial_conditions, forecast_values], dim=0)

        return target

    def __call__(
        self, sample: TimeSeriesSample
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return {"inputs": self.inputs(sample), "target": self.target(sample)}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(self.inputs(sample)["x"].shape[0])

    def n_outputs(self, sample: TimeSeriesSample) -> int:
        return int(self.target(sample).shape[0])


class CNN1DFeatParamICForecastAdapter:
    """
    Adapter for CNN1D with features parameter and initial conditions and forecast values prediction.
    Compatible with `model.forward(x, x_feat)`.
    """

    feature_names = ["is_steady_state", "steady_state_mean"]

    def inputs(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
        def flatten_nested_lists(values: Any) -> list[Any]:
            flat = []
            for v in values:
                if isinstance(v, list):
                    flat.extend(flatten_nested_lists(v))
                else:
                    flat.append(v)
            return flat

        x = torch.from_numpy(sample.time_series.copy()).float()
        feat_lst: list[Any] = flatten_nested_lists(
            [sample.features[name] for name in self.feature_names]
        )
        features = np.array([np.nan if v is None else v for v in feat_lst], dtype=float)

        x_feat = torch.from_numpy(features).float().flatten()
        return {"x": x, "x_feat": x_feat}

    def target(self, sample: TimeSeriesSample) -> torch.Tensor:
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        forecast_values = torch.from_numpy(np.array(sample.features["forecast_values"])).float()
        forecast_values = forecast_values.flatten()
        target = torch.cat([parameters, initial_conditions, forecast_values], dim=0)

        return target

    def __call__(
        self, sample: TimeSeriesSample
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return {"inputs": self.inputs(sample), "target": self.target(sample)}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(self.inputs(sample)["x"].shape[0])

    def n_outputs(self, sample: TimeSeriesSample) -> int:
        return int(self.target(sample).shape[0])


class InversePINNAdapter:
    """
    Adapter for inverse PINN model that maps trajectories to parameter and initial condition estimates.

    Inputs:
        - trajectory: Observed time series [u, a] of shape (2n, T)
        - time_grid: Time points of shape (T,)

    Targets:
        - parameter_ic: Concatenated parameter vector and initial conditions of shape (2n + 2n² + 2n,)
    """

    def __call__(
        self, sample: TimeSeriesSample
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        x = torch.from_numpy(sample.time_series.copy()).float()
        time_grid = torch.from_numpy(sample.time_points.copy()).float()
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        target = torch.cat([parameters, initial_conditions], dim=0)

        return {"inputs": {"x": x, "t": time_grid}, "target": target}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(sample.time_series.shape[0])

    def n_outputs(self, sample: TimeSeriesSample) -> int:
        return int(sample.parameters.values.shape[0] + sample.initial_conditions.values.shape[0])
