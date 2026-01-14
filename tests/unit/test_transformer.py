"""
Unit tests for data transformer classes.
"""

from collections.abc import Callable, Iterable

import numpy as np
import pytest

from preference_dynamics.data.transformer import (
    CyclesFeature,
    DataTransformer,
    DeleteSampleTransformer,
    InitialValueFeature,
    PeaksFeature,
    SampleGroupNormalizer,
    SampleGroupStdNormalizer,
    SampleNormalizer,
    ShortenTimeSeriesTransformer,
    SteadyStateFeature,
)
from preference_dynamics.schemas import TimeSeriesSample


@pytest.mark.parametrize(
    ("remove_positions", "expected_ids"),
    [
        ([1, 3], [0, 2, 4]),
        ([0, 1, 2, 3, 4], []),
    ],
)
def test_delete_sample_transformer(
    random_time_series_n3: TimeSeriesSample,
    remove_positions: list[int],
    expected_ids: Iterable[int],
) -> None:
    samples = [random_time_series_n3.model_copy(update={"sample_id": f"id-{i}"}) for i in range(5)]
    transformer = DeleteSampleTransformer(remove_positions=remove_positions)

    result = transformer.transform(samples)

    assert [s.sample_id for s in result] == [f"id-{i}" for i in expected_ids]


def test_delete_sample_transformer_errors_on_empty() -> None:
    transformer = DeleteSampleTransformer(remove_positions=[0])
    with pytest.raises(IndexError):
        transformer.transform([])


def test_delete_sample_transformer_out_of_range(random_time_series_n3: TimeSeriesSample) -> None:
    transformer = DeleteSampleTransformer(remove_positions=[10])
    samples = [random_time_series_n3.model_copy()]

    with pytest.raises(IndexError):
        transformer.transform(samples)


@pytest.mark.parametrize(
    "transformer_cls",
    [SampleNormalizer, SampleGroupNormalizer, SampleGroupStdNormalizer],
)
def test_normalizers_apply_statistics_and_keep_metadata(
    transformer_cls: type[DataTransformer],
    random_time_series_n3: TimeSeriesSample,
) -> None:
    transformer = transformer_cls()
    samples = [random_time_series_n3.model_copy()]
    original_meta = samples[0].metadata.copy()

    result = transformer.transform(samples)

    normalized = result[0]
    assert normalized.statistics is not None
    assert normalized.statistics.means.shape == (2 * normalized.n_actions,)
    assert normalized.statistics.stds.shape == (2 * normalized.n_actions,)
    np.testing.assert_array_equal(normalized.time_points, samples[0].time_points)
    assert normalized.metadata == original_meta


def test_sample_normalizer_handles_constant_series(random_time_series_n3: TimeSeriesSample) -> None:
    constant = random_time_series_n3.model_copy(
        update={"time_series": np.ones_like(random_time_series_n3.time_series)}
    )
    normalized = SampleNormalizer().transform([constant])[0]
    assert np.allclose(normalized.time_series, 0.0, atol=1e-8)


def test_sample_group_normalizer_handles_constant_groups(
    random_time_series_n3: TimeSeriesSample,
) -> None:
    desires = np.full_like(random_time_series_n3.desires, 2.0)
    efforts = np.full_like(random_time_series_n3.efforts, 3.0)
    constant = random_time_series_n3.model_copy(
        update={"time_series": np.concatenate([desires, efforts])}
    )
    normalized = SampleGroupNormalizer().transform([constant])[0]
    assert np.allclose(normalized.desires, 0.0, atol=1e-8)
    assert np.allclose(normalized.efforts, 0.0, atol=1e-8)


def test_sample_group_std_normalizer_preserves_zero_point(
    random_time_series_n3: TimeSeriesSample,
) -> None:
    """Test that SampleGroupStdNormalizer preserves zero point (doesn't subtract mean)."""
    transformer = SampleGroupStdNormalizer()
    ts = random_time_series_n3.model_copy()
    ts.time_series[:, :10] = 0.0
    samples = [ts]
    original_desires = samples[0].desires.copy()
    original_efforts = samples[0].efforts.copy()

    result = transformer.transform(samples)
    normalized = result[0]

    zero_mask_desires = original_desires == 0.0
    zero_mask_efforts = original_efforts == 0.0
    if np.any(zero_mask_desires):
        assert np.allclose(normalized.desires[zero_mask_desires], 0.0, atol=1e-8)
    if np.any(zero_mask_efforts):
        assert np.allclose(normalized.efforts[zero_mask_efforts], 0.0, atol=1e-8)

    assert np.allclose(normalized.statistics.means, 0.0, atol=1e-8)


def test_sample_group_std_normalizer_normalizes_std_to_one(
    random_time_series_n3: TimeSeriesSample,
) -> None:
    """Test that SampleGroupStdNormalizer normalizes std to approximately 1."""
    transformer = SampleGroupStdNormalizer()
    samples = [random_time_series_n3.model_copy()]

    result = transformer.transform(samples)
    normalized = result[0]

    desires_std = normalized.desires.std(axis=(0, 1))
    efforts_std = normalized.efforts.std(axis=(0, 1))
    assert np.allclose(desires_std, 1.0, atol=1e-6)
    assert np.allclose(efforts_std, 1.0, atol=1e-6)


def test_sample_group_std_normalizer_handles_constant_groups(
    random_time_series_n3: TimeSeriesSample,
) -> None:
    """Test that SampleGroupStdNormalizer handles constant series (std=0)."""
    desires = np.full_like(random_time_series_n3.desires, 2.0)
    efforts = np.full_like(random_time_series_n3.efforts, 3.0)
    constant = random_time_series_n3.model_copy(
        update={"time_series": np.concatenate([desires, efforts])}
    )
    normalized = SampleGroupStdNormalizer().transform([constant])[0]

    # With std=0, division by (std + 1e-8) should preserve the constant value scaled
    # The constant values should be preserved (scaled by 1/(1e-8) = 1e8)
    assert np.allclose(normalized.desires, 2.0 / 1e-8, atol=1e-6)
    assert np.allclose(normalized.efforts, 3.0 / 1e-8, atol=1e-6)

    assert np.allclose(normalized.statistics.means, 0.0, atol=1e-8)


def test_steady_state_feature_detects_steady_state(
    make_time_series_sample: Callable[..., TimeSeriesSample],
    steady_state_samples: dict[str, np.ndarray],
) -> None:
    feature = SteadyStateFeature(n_windows=4)
    steady = make_time_series_sample(steady_state_samples["steady"])
    drifting = make_time_series_sample(steady_state_samples["drifting"])
    short = make_time_series_sample(steady_state_samples["short"])

    transformed = feature.transform([steady, drifting, short])

    assert transformed[0].features["is_steady_state"] is True
    assert transformed[0].features["steady_state_mean"] is not None
    assert transformed[1].features["is_steady_state"] is False
    assert transformed[1].features["steady_state_mean"] == [None, None]
    assert transformed[2].features["is_steady_state"] is False


@pytest.mark.parametrize("tolerance", [0.0, -0.1])
def test_steady_state_feature_raises_on_negative_tolerances(tolerance: float) -> None:
    with pytest.raises(ValueError, match="Tolerances must be positive"):
        SteadyStateFeature(mean_abs_tol=tolerance)


def test_initial_value_feature_sets_first_timestep(
    make_time_series_sample: Callable[..., TimeSeriesSample],
) -> None:
    time_series = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sample = make_time_series_sample(time_series, n_actions=1)
    feature = InitialValueFeature()

    transformed = feature.transform([sample])[0]

    assert transformed.features["initial_value"] == [1.0, 4.0]


# TODO: parametrize num_peaks=3,4,5,min,max: should take last n peaks, and pad left with 0
def test_peaks_feature_extracts_peaks_no_padding(
    peaks_sample: TimeSeriesSample,
) -> None:
    # num_peaks=4 as in fixture
    feature = PeaksFeature(prominence=0.5, num_peaks=4)
    transformed = feature.transform([peaks_sample])[0]

    assert "peak_indices" in transformed.features
    np.testing.assert_array_equal(
        transformed.features["peak_indices"], [[5, 12, 19, 26], [4, 11, 18, 25]]
    )
    np.testing.assert_array_equal(
        transformed.features["peak_heights"], [[2.5, 2.2, 2.4, 2.3], [1.8, 1.7, 1.9, 1.85]]
    )
    np.testing.assert_array_equal(transformed.features["peak_diffs_mean"], [7.0, 7.0])
    np.testing.assert_array_equal(transformed.features["peak_diffs_std"], [0.0, 0.0])
    np.testing.assert_array_equal(transformed.features["peak_diffs_min"], [7.0, 7.0])
    np.testing.assert_array_equal(transformed.features["peak_diffs_max"], [7.0, 7.0])
    np.testing.assert_array_equal(
        transformed.features["peak_times"], [[5, 12, 19, 26], [4, 11, 18, 25]]
    )


def test_peaks_feature_callable_num_peaks(
    peaks_sample: TimeSeriesSample,
) -> None:
    modified_ts = peaks_sample.time_series.copy()
    modified_ts[1, 25] = 0.0  # drop one peak on the second channel
    sample = peaks_sample.model_copy(update={"time_series": modified_ts})
    feature = PeaksFeature(prominence=0.5, num_peaks=min)

    transformed = feature.transform([sample])[0]

    assert transformed.features["peak_indices"] == [[5, 12, 19, 26], [4, 11, 18]]
    np.testing.assert_array_equal(
        transformed.features["peak_heights"], [[2.2, 2.4, 2.3], [1.8, 1.7, 1.9]]
    )
    np.testing.assert_array_equal(transformed.features["peak_diffs_mean"], [7.0, 7.0])
    np.testing.assert_array_equal(transformed.features["peak_diffs_std"], [0.0, 0.0])
    np.testing.assert_array_equal(transformed.features["peak_diffs_min"], [7.0, 7.0])
    np.testing.assert_array_equal(transformed.features["peak_diffs_max"], [7.0, 7.0])
    np.testing.assert_array_equal(
        transformed.features["peak_times"], [[12.0, 19.0, 26.0], [4.0, 11.0, 18.0]]
    )


def test_cycles_feature_requires_peaks(
    peaks_sample: TimeSeriesSample,
) -> None:
    assert peaks_sample.features.get("peak_indices") is None
    cycles_feature = CyclesFeature()
    with pytest.raises(ValueError, match="apply PeaksFeature first"):
        cycles_feature.transform([peaks_sample])


def test_cycles_feature_detects_limit_cycle(
    peaks_sample: TimeSeriesSample,
) -> None:
    peaks_feature = PeaksFeature(prominence=1.0, num_peaks=4)
    peaks_populated = peaks_feature.transform([peaks_sample])
    cycles_feature = CyclesFeature(tail_percent=1.0)

    transformed = cycles_feature.transform(peaks_populated)[0]

    assert transformed.features["is_limit_cycle"] is True
    np.testing.assert_array_almost_equal(
        transformed.features["limit_cycle_means"],
        [[0.34285714285714286, 0.31428571428571433], [0.2714285714285714, 0.24285714285714285]],
    )
    np.testing.assert_array_almost_equal(
        transformed.features["limit_cycle_mean"], [0.3285714285714286, 0.2571428571428571]
    )
    np.testing.assert_array_equal(transformed.features["limit_cycle_diffs"], [[7, 7], [7, 7]])
    np.testing.assert_array_almost_equal(transformed.features["limit_cycle_mean_diff"], [7.0, 7.0])


class TestShortenTimeSeriesTransformer:
    """Test suite for ShortenTimeSeriesTransformer."""

    @pytest.mark.parametrize("n_steps", [1, 2, 5])
    def test_shortens_time_series_by_n_steps(
        self, n_steps: int, make_time_series_sample: Callable[..., TimeSeriesSample]
    ) -> None:
        """Test transformer shortens time series and extracts forecast values correctly."""
        n_channels = 4
        original_length = 20
        time_series = np.random.randn(n_channels, original_length)
        time_points = np.linspace(0.0, original_length - 1, original_length)
        sample = make_time_series_sample(time_series, time_points=time_points, n_actions=2)

        transformer = ShortenTimeSeriesTransformer(n_steps=n_steps)
        transformed = transformer.transform([sample])[0]

        expected_length = original_length - n_steps
        assert transformed.time_series.shape == (n_channels, expected_length)
        assert len(transformed.time_points) == expected_length
        assert transformed.config.solver.n_time_points == expected_length
        assert transformed.config.solver.time_span[1] == transformed.time_points[-1]

        forecast_values = np.array(transformed.features["forecast_values"])
        assert forecast_values.shape == (n_channels, n_steps)
        np.testing.assert_array_equal(forecast_values, time_series[:, -n_steps:])

        TimeSeriesSample.model_validate(transformed)

    def test_multiple_samples(
        self, make_time_series_sample: Callable[..., TimeSeriesSample]
    ) -> None:
        """Test transformer handles multiple samples correctly."""
        samples = [make_time_series_sample(np.random.randn(4, 20), n_actions=2) for _ in range(3)]

        transformer = ShortenTimeSeriesTransformer(n_steps=2)
        transformed = transformer.transform(samples)

        assert len(transformed) == 3
        for s in transformed:
            assert s.time_series.shape[1] == 18
            assert len(s.time_points) == 18
            forecast_values = np.array(s.features["forecast_values"])
            assert forecast_values.shape == (4, 2)
            TimeSeriesSample.model_validate(s)

    @pytest.mark.parametrize("n_steps", [0, -1])
    def test_raises_on_invalid_n_steps(self, n_steps: int) -> None:
        """Test transformer raises ValueError for invalid n_steps."""
        with pytest.raises(ValueError, match="n_steps must be positive"):
            ShortenTimeSeriesTransformer(n_steps=n_steps)
