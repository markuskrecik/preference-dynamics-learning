"""
Unit tests for model adapters.
"""

import numpy as np
import pytest
import torch

from preference_dynamics.data.adapters import (
    CNN1DParamAdapter,
    CNN1DParamICAdapter,
    CNN1DParamICForecastAdapter,
)


class TestCNN1DParamAdapter:
    """Test suite for CNN1DParamAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_adapter_transforms_sample_correctly(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter correctly transforms sample with correct structure, shapes, dtypes, and values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = CNN1DParamAdapter()
        result = adapter(sample)

        assert "inputs" in result
        assert "target" in result
        assert isinstance(result["inputs"], dict)
        assert "x" in result["inputs"]

        x = result["inputs"]["x"]
        target = result["target"]

        assert isinstance(x, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert x.dtype == torch.float32
        assert target.dtype == torch.float32
        assert x.shape == sample.time_series.shape
        assert target.shape == sample.parameters.values.shape

        np.testing.assert_allclose(x.numpy(), sample.time_series)
        np.testing.assert_allclose(target.numpy(), sample.parameters.values)

        assert adapter.n_inputs(sample) == sample.time_series.shape[0]
        assert adapter.n_outputs(sample) == sample.parameters.values.shape[0]

    @pytest.mark.parametrize("seq_len", [2, 100, 1000])
    def test_adapter_handles_different_sequence_lengths(
        self, seq_len: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various sequence lengths."""
        time_series = np.random.randn(4, seq_len)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        adapter = CNN1DParamAdapter()
        result = adapter(sample)

        assert result["inputs"]["x"].shape[1] == seq_len
        assert adapter.n_inputs(sample) == 4


class TestCNN1DParamICAdapter:
    """Test suite for CNN1DParamICAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3, 4, 5])
    def test_adapter_transforms_sample_correctly(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter correctly transforms sample with correct structure, shapes, dtypes, values, and concatenation order."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        adapter = CNN1DParamICAdapter()
        result = adapter(sample)

        assert "inputs" in result
        assert "target" in result
        assert isinstance(result["inputs"], dict)
        assert "x" in result["inputs"]

        x = result["inputs"]["x"]
        target = result["target"]

        assert isinstance(x, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert x.dtype == torch.float32
        assert target.dtype == torch.float32
        assert x.shape == sample.time_series.shape

        expected_target_len = (
            sample.parameters.values.shape[0] + sample.initial_conditions.values.shape[0]
        )
        assert target.shape[0] == expected_target_len

        np.testing.assert_allclose(x.numpy(), sample.time_series)

        expected_target = np.concatenate(
            [sample.parameters.values, sample.initial_conditions.values]
        )
        np.testing.assert_allclose(target.numpy(), expected_target)

        params_len = sample.parameters.values.shape[0]
        target_params = target[:params_len].numpy()
        target_ic = target[params_len:].numpy()
        np.testing.assert_allclose(target_params, sample.parameters.values)
        np.testing.assert_allclose(target_ic, sample.initial_conditions.values)

        assert adapter.n_inputs(sample) == sample.time_series.shape[0]
        assert adapter.n_outputs(sample) == expected_target_len

    @pytest.mark.parametrize("seq_len", [2, 100, 1000])
    def test_adapter_handles_different_sequence_lengths(
        self, seq_len: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various sequence lengths."""
        time_series = np.random.randn(4, seq_len)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        adapter = CNN1DParamICAdapter()
        result = adapter(sample)

        assert result["inputs"]["x"].shape[1] == seq_len
        assert adapter.n_inputs(sample) == 4


class TestCNN1DParamICForecastAdapter:
    """Test suite for CNN1DParamICForecastAdapter."""

    @pytest.mark.parametrize("n_actions", [1, 2, 3])
    def test_adapter_transforms_sample_correctly(
        self, n_actions: int, make_time_series_sample
    ) -> None:
        """Test adapter correctly transforms sample with forecast values."""
        time_series = np.random.randn(2 * n_actions, 50)
        time_series[n_actions:, :] = np.abs(time_series[n_actions:, :])
        sample = make_time_series_sample(time_series, n_actions=n_actions)
        n_steps = 3
        forecast_values = np.random.randn(2 * n_actions, n_steps)
        sample.features["forecast_values"] = forecast_values.tolist()

        adapter = CNN1DParamICForecastAdapter()
        result = adapter(sample)

        assert "inputs" in result
        assert "target" in result
        assert isinstance(result["inputs"], dict)
        assert "x" in result["inputs"]

        x = result["inputs"]["x"]
        target = result["target"]

        assert isinstance(x, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert x.dtype == torch.float32
        assert target.dtype == torch.float32
        assert x.shape == sample.time_series.shape

        expected_target_len = (
            sample.parameters.values.shape[0]
            + sample.initial_conditions.values.shape[0]
            + forecast_values.size
        )
        assert target.shape[0] == expected_target_len

        np.testing.assert_allclose(x.numpy(), sample.time_series)

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

        assert adapter.n_inputs(sample) == sample.time_series.shape[0]
        assert adapter.n_outputs(sample) == expected_target_len

    @pytest.mark.parametrize("seq_len", [10, 50, 100])
    def test_adapter_handles_different_sequence_lengths(
        self, seq_len: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various sequence lengths."""
        time_series = np.random.randn(4, seq_len)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        sample.features["forecast_values"] = np.random.randn(4, 2).tolist()

        adapter = CNN1DParamICForecastAdapter()
        result = adapter(sample)

        assert result["inputs"]["x"].shape[1] == seq_len
        assert adapter.n_inputs(sample) == 4

    @pytest.mark.parametrize("n_steps", [1, 2, 5])
    def test_adapter_handles_different_forecast_lengths(
        self, n_steps: int, make_time_series_sample
    ) -> None:
        """Test adapter works with various forecast value lengths."""
        time_series = np.random.randn(4, 30)
        time_series[2:, :] = np.abs(time_series[2:, :])
        sample = make_time_series_sample(time_series, n_actions=2)
        forecast_values = np.random.randn(4, n_steps)
        sample.features["forecast_values"] = forecast_values.tolist()

        adapter = CNN1DParamICForecastAdapter()
        result = adapter(sample)

        expected_target_len = (
            sample.parameters.values.shape[0]
            + sample.initial_conditions.values.shape[0]
            + forecast_values.size
        )
        assert result["target"].shape[0] == expected_target_len
        assert adapter.n_outputs(sample) == expected_target_len
