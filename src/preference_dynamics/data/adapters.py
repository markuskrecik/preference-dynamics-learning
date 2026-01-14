"""
Model Adapters for flexible model inputs and targets specification from arbitrary data sources.
Ingested by torch.utils.data.Dataset.
"""

from typing import Any, Protocol

import numpy as np
import torch

from preference_dynamics.schemas import TimeSeriesSample
from preference_dynamics.solver.torch_equations import preference_dynamics_rhs_torch


class InputAdapter(Protocol):
    """
    Adapter for model input and output.
    """

    def get_inputs(self, sample: Any) -> dict[str, torch.Tensor]: ...

    def n_inputs(self, sample: Any) -> int | tuple[int, ...]: ...


class TargetAdapter(Protocol):
    def get_targets(self, sample: Any) -> torch.Tensor | dict[str, torch.Tensor]: ...

    def n_targets(self, sample: Any) -> int | tuple[int, ...]: ...


class StateInputAdapter:
    """
    Adapter for state input. Compatible with `model.forward(x)`.
    """

    def get_inputs(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(sample.time_series.copy()).float()
        return {"x": x}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(self.get_inputs(sample)["x"].shape[0])


class StateFeatureInputAdapter:
    """
    Adapter for state and feature input. Compatible with `model.forward(x, x_feat)`.
    """

    feature_names = ["is_steady_state", "steady_state_mean"]

    def get_inputs(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
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

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(self.get_inputs(sample)["x"].shape[0])


class StateTimeInputAdapter:
    """
    Adapter for state and time input. Compatible with `model.forward(x, t)`.
    """

    def get_inputs(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(sample.time_series.copy()).float()
        time_points = torch.from_numpy(sample.time_points.copy()).float()
        return {"x": x, "t": time_points}

    def n_inputs(self, sample: TimeSeriesSample) -> int:
        return int(self.get_inputs(sample)["x"].shape[0])


class ParamICTimeInputAdapter:
    """
    Adapter for parameter, initial conditions, and time input.

    Returns:
        dict with keys "params", "ic", "t"
    """

    def get_inputs(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        time_points = torch.from_numpy(sample.time_points.copy()).float()
        return {"params": parameters, "ic": initial_conditions, "t": time_points}

    def n_inputs(self, sample: TimeSeriesSample) -> tuple[int, ...]:
        inputs = self.get_inputs(sample)
        return tuple(int(i.shape[0]) for i in inputs.values())


class ParameterTargetAdapter:
    """
    Adapter for parameter target.
    """

    def get_targets(self, sample: TimeSeriesSample) -> torch.Tensor:
        target = torch.from_numpy(sample.parameters.values.copy()).float()
        return target

    def n_targets(self, sample: TimeSeriesSample) -> int:
        return int(self.get_targets(sample).shape[0])


class ParameterICTargetAdapter:
    """
    Adapter for parameter and initial conditions target.
    """

    def get_targets(self, sample: TimeSeriesSample) -> torch.Tensor:
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        target = torch.cat([parameters, initial_conditions], dim=0)
        return target

    def n_targets(self, sample: TimeSeriesSample) -> int:
        return int(self.get_targets(sample).shape[0])


class ForecastTargetAdapter:
    """
    Adapter for forecast values target.
    """

    def get_targets(self, sample: TimeSeriesSample) -> torch.Tensor:
        forecast_values = torch.from_numpy(np.array(sample.features["forecast_values"])).float()
        return forecast_values.flatten()

    def n_targets(self, sample: TimeSeriesSample) -> int:
        return int(self.get_targets(sample).shape[0])


class ParameterICForecastTargetAdapter:
    """
    Adapter for parameter, initial conditions and forecast values target.
    """

    def get_targets(self, sample: TimeSeriesSample) -> torch.Tensor:
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        forecast_values = torch.from_numpy(np.array(sample.features["forecast_values"])).float()
        forecast_values = forecast_values.flatten()
        target = torch.cat([parameters, initial_conditions, forecast_values], dim=0)
        return target

    def n_targets(self, sample: TimeSeriesSample) -> int:
        return int(self.get_targets(sample).shape[0])


class ParamICStateTimeDerivativeTargetAdapter:
    """
    Adapter for parameter, initial conditions, state, time, and time derivative targets.

    Returns:
        dict with keys "params", "ic", "x", "d_latent_x", "t"
    """

    def get_targets(self, sample: TimeSeriesSample) -> dict[str, torch.Tensor]:
        params = torch.from_numpy(sample.parameters.values.copy()).float()
        initial_conditions = torch.from_numpy(sample.initial_conditions.values.copy()).float()
        state = torch.from_numpy(sample.time_series.copy()).float()  # (2n, T)
        time_points = torch.from_numpy(sample.time_points.copy()).float()

        d_latent_x = preference_dynamics_rhs_torch(
            t=time_points,
            state=state,
            params=params,
        )
        return {
            "params": params,
            "ic": initial_conditions,
            "x": state,
            "d_latent_x": d_latent_x,
            "t": time_points,
        }

    def n_targets(self, sample: TimeSeriesSample) -> tuple[int, ...]:
        targets = self.get_targets(sample)
        return tuple(int(t.shape[0]) for t in targets.values())
