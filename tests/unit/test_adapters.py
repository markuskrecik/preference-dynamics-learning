"""
Unit tests for model adapters.
"""

import numpy as np
import pytest
import torch

from preference_dynamics.data.adapters import (
    ParameterICForecastTargetAdapter,
    ParameterICTargetAdapter,
    ParameterTargetAdapter,
    StateFeatureInputAdapter,
    StateInputAdapter,
)


class TestStateInputAdapter:
    """Test suite for StateInputAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_get_inputs_returns_correct_structure(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter returns correct structure, shapes, dtypes, and values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = StateInputAdapter()

        inputs = adapter.get_inputs(sample)

        assert isinstance(inputs, dict)
        assert "x" in inputs

        x = inputs["x"]

        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32
        assert x.shape == sample.time_series.shape

        np.testing.assert_allclose(x.numpy(), sample.time_series)

    @pytest.mark.parametrize("seq_len", [2, 100, 1000])
    def test_get_inputs_handles_different_sequence_lengths(
        self, seq_len: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various sequence lengths."""
        time_series = np.random.randn(4, seq_len)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        adapter = StateInputAdapter()
        inputs = adapter.get_inputs(sample)

        assert inputs["x"].shape[1] == seq_len

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_n_inputs_returns_correct_count(self, n_actions: int, make_time_series_sample) -> None:
        """Test n_inputs returns correct number of input channels."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = StateInputAdapter()

        assert adapter.n_inputs(sample) == sample.time_series.shape[0]


class TestStateFeatureInputAdapter:
    """Test suite for StateFeatureInputAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_get_inputs_returns_correct_structure(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter returns correct structure with x and x_feat."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        sample.features["is_steady_state"] = [True, False]
        sample.features["steady_state_mean"] = [1.0, 2.0]
        adapter = StateFeatureInputAdapter()

        inputs = adapter.get_inputs(sample)

        assert isinstance(inputs, dict)
        assert "x" in inputs
        assert "x_feat" in inputs

        x = inputs["x"]
        x_feat = inputs["x_feat"]

        assert isinstance(x, torch.Tensor)
        assert isinstance(x_feat, torch.Tensor)
        assert x.dtype == torch.float32
        assert x_feat.dtype == torch.float32
        assert x.shape == sample.time_series.shape

        np.testing.assert_allclose(x.numpy(), sample.time_series)

    def test_get_inputs_handles_none_features(self, make_time_series_sample) -> None:
        """Test adapter handles None values in features."""
        time_series = np.random.randn(4, 50)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        sample.features["is_steady_state"] = None
        sample.features["steady_state_mean"] = None
        adapter = StateFeatureInputAdapter()

        inputs = adapter.get_inputs(sample)

        assert "x_feat" in inputs
        assert isinstance(inputs["x_feat"], torch.Tensor)

    def test_get_inputs_handles_nested_lists(self, make_time_series_sample) -> None:
        """Test adapter handles nested lists in features."""
        time_series = np.random.randn(4, 50)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        sample.features["is_steady_state"] = [[True], [False]]
        sample.features["steady_state_mean"] = [[1.0], [2.0]]
        adapter = StateFeatureInputAdapter()

        inputs = adapter.get_inputs(sample)

        assert "x_feat" in inputs
        assert isinstance(inputs["x_feat"], torch.Tensor)

    @pytest.mark.parametrize("seq_len", [2, 100, 1000])
    def test_get_inputs_handles_different_sequence_lengths(
        self, seq_len: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various sequence lengths."""
        time_series = np.random.randn(4, seq_len)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        sample.features["is_steady_state"] = [True, False]
        sample.features["steady_state_mean"] = [1.0, 2.0]
        adapter = StateFeatureInputAdapter()
        inputs = adapter.get_inputs(sample)

        assert inputs["x"].shape[1] == seq_len

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_n_inputs_returns_correct_count(self, n_actions: int, make_time_series_sample) -> None:
        """Test n_inputs returns correct number of input channels."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        sample.features["is_steady_state"] = [True, False]
        sample.features["steady_state_mean"] = [1.0, 2.0]
        adapter = StateFeatureInputAdapter()

        assert adapter.n_inputs(sample) == sample.time_series.shape[0]


class TestParameterTargetAdapter:
    """Test suite for ParameterTargetAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_get_targets_returns_correct_structure(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter returns correct structure, shapes, dtypes, and values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = ParameterTargetAdapter()

        target = adapter.get_targets(sample)

        assert isinstance(target, torch.Tensor)
        assert target.dtype == torch.float32
        assert target.shape == sample.parameters.values.shape

        np.testing.assert_allclose(target.numpy(), sample.parameters.values)

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_n_targets_returns_correct_count(self, n_actions: int, make_time_series_sample) -> None:
        """Test n_targets returns correct number of target values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = ParameterTargetAdapter()

        assert adapter.n_targets(sample) == sample.parameters.values.shape[0]


class TestParameterICTargetAdapter:
    """Test suite for ParameterICTargetAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3, 4, 5])
    def test_get_targets_returns_correct_structure(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter returns correct structure, shapes, dtypes, values, and concatenation order."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = ParameterICTargetAdapter()

        target = adapter.get_targets(sample)

        assert isinstance(target, torch.Tensor)
        assert target.dtype == torch.float32

        expected_target_len = (
            sample.parameters.values.shape[0] + sample.initial_conditions.values.shape[0]
        )
        assert target.shape[0] == expected_target_len

        expected_target = np.concatenate(
            [sample.parameters.values, sample.initial_conditions.values]
        )
        np.testing.assert_allclose(target.numpy(), expected_target)

        params_len = sample.parameters.values.shape[0]
        target_params = target[:params_len].numpy()
        target_ic = target[params_len:].numpy()
        np.testing.assert_allclose(target_params, sample.parameters.values)
        np.testing.assert_allclose(target_ic, sample.initial_conditions.values)

    @pytest.mark.parametrize("n_actions", [1, 2, 3, 4, 5])
    def test_n_targets_returns_correct_count(self, n_actions: int, make_time_series_sample) -> None:
        """Test n_targets returns correct number of target values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = ParameterICTargetAdapter()

        expected_target_len = (
            sample.parameters.values.shape[0] + sample.initial_conditions.values.shape[0]
        )
        assert adapter.n_targets(sample) == expected_target_len


class TestParameterICForecastTargetAdapter:
    """Test suite for ParameterICForecastTargetAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_get_targets_returns_correct_structure(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter returns correct structure with forecast values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        n_steps = 3
        forecast_values = np.random.randn(2 * n_actions, n_steps)
        sample.features["forecast_values"] = forecast_values.tolist()

        adapter = ParameterICForecastTargetAdapter()
        target = adapter.get_targets(sample)

        assert isinstance(target, torch.Tensor)
        assert target.dtype == torch.float32

        expected_target_len = (
            sample.parameters.values.shape[0]
            + sample.initial_conditions.values.shape[0]
            + forecast_values.size
        )
        assert target.shape[0] == expected_target_len

        expected_target = np.concatenate(
            [
                sample.parameters.values,
                sample.initial_conditions.values,
                forecast_values.flatten(),
            ]
        )
        np.testing.assert_allclose(target.numpy(), expected_target)

        params_len = sample.parameters.values.shape[0]
        ic_len = sample.initial_conditions.values.shape[0]
        target_params = target[:params_len].numpy()
        target_ic = target[params_len : params_len + ic_len].numpy()
        target_forecast = target[params_len + ic_len :].numpy()

        np.testing.assert_allclose(target_params, sample.parameters.values)
        np.testing.assert_allclose(target_ic, sample.initial_conditions.values)
        np.testing.assert_allclose(target_forecast, forecast_values.flatten())

    @pytest.mark.parametrize("n_steps", [1, 2, 5])
    def test_get_targets_handles_different_forecast_lengths(
        self, n_steps: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various forecast value lengths."""
        time_series = np.random.randn(4, 30)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        forecast_values = np.random.randn(4, n_steps)
        sample.features["forecast_values"] = forecast_values.tolist()

        adapter = ParameterICForecastTargetAdapter()
        target = adapter.get_targets(sample)

        expected_target_len = (
            sample.parameters.values.shape[0]
            + sample.initial_conditions.values.shape[0]
            + forecast_values.size
        )
        assert target.shape[0] == expected_target_len

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_n_targets_returns_correct_count(self, n_actions: int, make_time_series_sample) -> None:
        """Test n_targets returns correct number of target values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        n_steps = 3
        forecast_values = np.random.randn(2 * n_actions, n_steps)
        sample.features["forecast_values"] = forecast_values.tolist()

        adapter = ParameterICForecastTargetAdapter()

        expected_target_len = (
            sample.parameters.values.shape[0]
            + sample.initial_conditions.values.shape[0]
            + forecast_values.size
        )
        assert adapter.n_targets(sample) == expected_target_len
