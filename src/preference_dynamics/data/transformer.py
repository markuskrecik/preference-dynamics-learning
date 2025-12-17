"""
Data transformers for preference dynamics time series data.
"""

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import numpy as np
from numpydantic import NDArray

from preference_dynamics.schemas import SampleStatistics, TimeSeriesSample


class DataTransformer(Protocol):
    """
    Protocol for DataTransformers that process raw data into processed data.
    pre_split: Specifies whether the transformer must be applied before or after train-val-test split.
    """

    pre_split: bool

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]: ...


class DeleteSampleTransformer:
    """
    Transformer which deletes samples at specified positions.

    Removes samples at specified indices from the dataset. Applied before
    train-val-test split to ensure consistent removal across splits.
    """

    pre_split: bool = True

    def __init__(self, remove_positions: Sequence[int]):
        self.remove_positions = remove_positions

    def transform(self, samples: Sequence[TimeSeriesSample]) -> list[TimeSeriesSample]:
        return list(np.delete(np.asarray(samples), self.remove_positions))


class SampleNormalizer:
    """
    Transformer which normalizes each sample independently.

    Normalizes using the mean and std computed across all channels and time steps
    in each sample. Updates sample.time_series in-place and sets sample.statistics.
    """

    pre_split: bool = False

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            n = s.n_actions
            X = s.time_series

            mean = X.mean(axis=(0, 1), keepdims=True)
            std = X.std(axis=(0, 1), keepdims=True)

            X_norm = (X - mean) / (std + 1e-8)
            means = mean.repeat(2 * n)
            stds = std.repeat(2 * n)

            statistics = SampleStatistics(method="sample_normalization", means=means, stds=stds)

            s.time_series = X_norm
            s.statistics = statistics

        return samples


class SampleGroupNormalizer:
    """
    Transformer which normalizes each sample for each variable group (desires, efforts) independently.

    Normalizes across desire and effort channels separately using their respective means and stds
    across all time steps. Updates sample.time_series in-place and sets sample.statistics.
    """

    pre_split: bool = False

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            n = s.n_actions
            means = []
            stds = []
            X_norms = []
            for X in [s.desires, s.efforts]:
                mean = X.mean(axis=(0, 1), keepdims=True)
                std = X.std(axis=(0, 1), keepdims=True)
                X_norm = (X - mean) / (std + 1e-8)
                means.append(mean.repeat(n))
                stds.append(std.repeat(n))
                X_norms.append(X_norm)

            statistics = SampleStatistics(
                method="sample_group_normalization",
                means=np.concatenate(means),
                stds=np.concatenate(stds),
            )

            s.time_series = np.concatenate(X_norms)
            s.statistics = statistics

        return samples


class SampleGroupStdNormalizer:
    """
    Transformer which normalizes only by std for each sample for each variable group (desires, efforts) independently.

    Normalizes only by std (without subtracting mean) across desire and effort channels separately.
    This preserves the zero point of the original time series while scaling std to 1.
    Updates sample.time_series in-place and sets sample.statistics with means=0.
    """

    pre_split: bool = False

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            n = s.n_actions
            means = []
            stds = []
            X_norms = []
            for X in [s.desires, s.efforts]:
                std = X.std(axis=(0, 1), keepdims=True)
                X_norm = X / (std + 1e-8)
                means.append(np.zeros(n))  # zero means since we don't subtract
                stds.append(std.repeat(n))
                X_norms.append(X_norm)

            statistics = SampleStatistics(
                method="sample_group_std_normalization",
                means=np.concatenate(means),
                stds=np.concatenate(stds),
            )

            s.time_series = np.concatenate(X_norms)
            s.statistics = statistics

        return samples


class InitialValueFeature:
    """
    Transformer which extracts the first time step values (time_series[:, 0]).
    Note that these are observed values (u, a), not the latent state (v, m).
    They equal initial conditions only if efforts > 0 and desires > μ.
    """

    pre_split: bool = True  # Ensures to get initial values before normalization

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            initial_value: NDArray[Literal["*"], float] = s.time_series[:, 0]
            # tolist(), because pydantic does not know how to serialize numpy arrays without explicit model.
            s.features["initial_value"] = initial_value.tolist()
        return samples


class SteadyStateFeature:
    """
    Transformer which detects if the time series is in a steady state.
    It first checks if there is variation within non-overlapping windows over the last half of the time series.
    If not, it check if there is a drift in means of the windows.
    If there is none, the time series is in a steady state.

    Args:
        n_windows: number of non-overlapping windows (default: 4)
        mean_abs_tol: absolute tolerance for mean difference check (default: 0.01)
        mean_rel_tol: relative tolerance for mean difference check (default: 0.02)
        std_abs_tol: absolute tolerance for within-window std check (default: 0.02)
        std_rel_tol: relative tolerance for within-window std check (default: 0.001)

    Returns:
        list[TimeSeriesSample]: Updates sample.features dict with `is_steady_state` and `steady_state_mean` features
    """

    pre_split: bool = True  # Ensures to get steady state means before normalization

    def __init__(
        self,
        n_windows: int = 4,
        mean_abs_tol: float = 1e-2,
        mean_rel_tol: float = 0.02,
        std_abs_tol: float = 0.02,
        std_rel_tol: float = 1e-3,
    ):
        if any(np.array([mean_abs_tol, mean_rel_tol, std_abs_tol, std_rel_tol]) <= 0.0):
            raise ValueError("Tolerances must be positive.")
        self.n_windows = n_windows
        self.mean_abs_tol = mean_abs_tol
        self.mean_rel_tol = mean_rel_tol
        self.std_abs_tol = std_abs_tol
        self.std_rel_tol = std_rel_tol

    def detect_steady_state(
        self,
        X: NDArray[Literal["*", "*"], float],  # shape: (n_channels, T)
    ) -> tuple[bool, list[float] | None]:
        """
        Detect if time series is in steady state by analyzing last half of time series.

        Analyzes non-overlapping windows from the end of the time series.
        Checks if within-window std is below threshold, then checks if mean
        differences between windows are below threshold.

        Args:
            X: (n_channels, T) time series

        Returns:
            Tuple of:
            - is_steady: bool, True if the time series is in a steady state
            - last_mean: list[float] | None, mean values of the last steady window per channel if steady, None otherwise
        """

        tail_len = max(1, X.shape[1] // 2)  # don't need to check the whole series
        tail = X[:, -tail_len:]

        window_size = tail_len // self.n_windows
        if window_size == 0:
            return False, None

        means = []
        flat_windows = []
        for i in range(self.n_windows):
            end = tail_len - i * window_size
            start = end - window_size
            window = tail[:, start:end]
            mean = window.mean(axis=1)
            within_std = window.std(axis=1)
            std_thr = self.std_abs_tol + self.std_rel_tol * np.abs(mean)
            if np.all(within_std <= std_thr):
                flat_windows.append(i)
            else:
                break
            means.append(mean)

        if len(flat_windows) == 0:
            return False, None

        last_mean = means[0]  # more robust to noise than last value

        means_np: NDArray[Literal["*", "*"], float] = np.stack(means, axis=0)
        means_diff = np.abs(means_np[flat_windows] - last_mean)
        mean_thr = self.mean_abs_tol + self.mean_rel_tol * np.abs(last_mean)
        is_steady = np.all(means_diff <= mean_thr)

        return bool(is_steady), last_mean.tolist()

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            is_steady, last_mean = self.detect_steady_state(s.time_series)

            if is_steady:
                s.features["is_steady_state"] = True
                s.features["steady_state_mean"] = last_mean
            else:
                n = s.time_series.shape[0]
                s.features["is_steady_state"] = False
                s.features["steady_state_mean"] = [None] * n

        return samples


class PeaksFeature:
    """
    Transformer which detects peaks in time series and extracts peak statistics.

    Detects peaks in each channel using scipy.signal.find_peaks with prominence threshold.
    Extracts peak times, peak delays, heights, and their statistics.

    Args:
        prominence: prominence threshold for peak detection (default: 1.0)
        num_peaks: number of peaks to extract per channel (padded left with 0),
            or callable (e.g., min, max) to use min/max number of peaks across channels (default: max)

    Updates sample.features dict inplace with `peak_*` features.
    """

    pre_split: bool = True  # Ensures to get peak heights before normalization

    def __init__(self, prominence: float = 1.0, num_peaks: int | Callable[[list[int]], int] = max):
        self.prominence = prominence
        self.num_peaks = num_peaks

    def get_last_n_padded(
        self, X: Sequence[Sequence[float | int]], n: int, value: float | int = 0
    ) -> NDArray[Literal["*"], float]:
        """
        Convert sequence of sequences X to rectangular array by padding left
        with `value` if shorter than `n`.

        Args:
            X: Sequence of sequences
            n: Target length
            value: Padding value (default: 0)

        Returns:
            Array of shape (len(X), n)
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return np.empty((len(X), 0))
        padded_X = []
        for x in X:
            x_np = np.asarray(x)
            x_np = np.pad(x_np[-n:], (max(0, n - len(x_np)), 0), constant_values=value)
            padded_X.append(x_np)
        return np.asarray(padded_X)

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        from scipy.signal import find_peaks  # noqa: PLC0415

        for s in samples:
            peak_indices = [find_peaks(ts, prominence=self.prominence)[0] for ts in s.time_series]
            if isinstance(self.num_peaks, int):
                peaks_len = self.num_peaks
            else:
                peaks_len = self.num_peaks([len(p) for p in peak_indices])
            peak_indices_padded = self.get_last_n_padded(peak_indices, peaks_len)

            s.features["peak_indices"] = [p.tolist() for p in peak_indices]

            if peaks_len > 1:
                peak_times = s.time_points[peak_indices_padded]
                peak_diffs = np.diff(peak_times)
                peak_heights = np.take_along_axis(
                    s.time_series, np.asarray(peak_indices_padded), axis=1
                )
                s.features["peak_times"] = peak_times.tolist()
                s.features["peak_diffs"] = peak_diffs.tolist()
                s.features["peak_diffs_mean"] = peak_diffs.mean(axis=1).tolist()
                s.features["peak_diffs_std"] = peak_diffs.std(axis=1).tolist()
                s.features["peak_diffs_min"] = peak_diffs.min(axis=1).tolist()
                s.features["peak_diffs_max"] = peak_diffs.max(axis=1).tolist()
                s.features["peak_heights"] = peak_heights.tolist()
            else:
                n = s.time_series.shape[0]
                s.features["peak_times"] = [[None]] * n
                s.features["peak_diffs"] = [[None]] * n
                s.features["peak_diffs_mean"] = [None] * n
                s.features["peak_diffs_std"] = [None] * n
                s.features["peak_diffs_min"] = [None] * n
                s.features["peak_diffs_max"] = [None] * n
                s.features["peak_heights"] = [[None]] * n
        return samples


class CyclesFeature:
    """
    Transformer which detects if the time series is in a limit cycle.

    Splits time series into cycles based on peak gaps, then checks if the last
    cycles have comparable means and time differences.
    Requires PeaksFeature to be applied first.

    Args:
        tail_percent: percentage of time series to use for detection (default: 0.5)
        mean_abs_tol: absolute tolerance for mean difference check (default: 0.5)
        mean_rel_tol: relative tolerance for mean difference check (default: 0.2)
        diff_abs_tol: absolute tolerance for time difference check (default: 0.5)
        diff_rel_tol: relative tolerance for time difference check (default: 0.02)
        q: quantile used to identify cycles (default: 0.75)
        gap_factor: fraction of quantile that defines cycle separation (default: 0.6)

    Updates sample.features dict inplace with `is_limit_cycle`, `limit_cycle_*` features.
    """

    pre_split: bool = False

    def __init__(
        self,
        tail_percent: float | None = None,
        mean_abs_tol: float = 0.5,
        mean_rel_tol: float = 0.2,
        diff_abs_tol: float = 0.5,
        diff_rel_tol: float = 0.02,
        q: float = 0.75,
        gap_factor: float = 0.6,
    ):
        self.tail_percent = tail_percent or 0.5
        self.mean_abs_tol = mean_abs_tol
        self.mean_rel_tol = mean_rel_tol
        self.diff_abs_tol = diff_abs_tol
        self.diff_rel_tol = diff_rel_tol
        self.q = q
        self.gap_factor = gap_factor

        if self.tail_percent < 0 or self.tail_percent > 1:
            raise ValueError("tail_percent must be between 0 and 1.")
        if any(
            np.array([self.mean_abs_tol, self.mean_rel_tol, self.diff_abs_tol, self.diff_rel_tol])
            <= 0.0
        ):
            raise ValueError("Tolerances must be positive.")
        if self.q < 0 or self.q > 1:
            raise ValueError("q must be between 0 and 1.")
        if self.gap_factor <= 0:
            raise ValueError("gap_factor must be positive.")

    def split_into_cycles(
        self, peaks: NDArray[Literal["*"], float | int], q: float = 0.75, gap_factor: float = 0.6
    ) -> list[NDArray[Literal["*"], float | int]]:
        """
        Split sorted peak indices into cycles based on gap detection.

        Identifies cycles by detecting large gaps between peaks. A cycle boundary
        occurs when the gap between consecutive peaks exceeds a threshold based
        on the quantile of gap sizes.

        | Data condition               | q    | gap_factor |
        | ---------------------------- | ---- | ---------- |
        | Clean, regular               | 0.7  | 0.6        |
        | Mild jitter                  | 0.8  | 0.6-0.7    |
        | Heavy jitter / missing peaks | 0.85 | 0.7-0.8    |

        Args:
            peaks: sorted peak indices
            q: quantile used to detect large gaps (default: 0.75)
            gap_factor: fraction of quantile that defines cycle separation (default: 0.6)

        Returns:
            list[np.ndarray]: list of peak indices for each cycle
        """
        peaks = np.asarray(peaks)
        if len(peaks) <= 1:
            return [peaks]

        d = np.diff(peaks)

        # robust estimate of large gaps
        large_gap_est = np.quantile(d, q)
        thr = gap_factor * large_gap_est

        starts = np.where(d >= thr)[0] + 1  # omits first cycle start at index 0
        return np.split(peaks, starts)

    def detect_limit_cycle(
        self,
        x: NDArray[Literal["*"], float | int],
        cycles: list[NDArray[Literal["*"], float | int]],
    ) -> tuple[
        bool, list[NDArray[Literal["*"], float]] | None, list[NDArray[Literal["*"], float]] | None
    ]:
        """
        Detect if a single time series ends up in a limit cycle.

        Analyzes cycles in the tail of the time series (based on tail_percent).
        Checks if the last cycles have comparable means and time differences.

        Args:
            x: time series array
            cycles: list of peak indices for each cycle

        Returns:
            Tuple of:
            - is_limit_cycle: True if the time series ends up in a limit cycle
            - means_limit_cycle: list of means of the last limit cycles or None
            - diffs_limit_cycle: list of time differences of the last limit cycles or None
        """

        if len(cycles) == 0:
            return False, None, None
        if any(len(c) == 0 for c in cycles):
            return False, None, None

        tail_start = int(x.shape[0] * (1.0 - self.tail_percent))
        starts = [c[0] for c in cycles if c[0] > tail_start]

        cycle_diffs = np.diff(starts)
        means = []
        for i in range(len(starts) - 1):
            window = x[starts[i] : starts[i + 1]]
            mean = window.mean()
            means.append(mean)
        means_np: NDArray[Literal["*"], float] = np.asarray(means)

        means_limit_cycle = []
        diffs_limit_cycle = []
        for i in range(len(means) - 1, 0, -1):
            mean_std = means_np[i:].std()
            diff_std = cycle_diffs[i:].std()
            mean_thr = self.mean_abs_tol + self.mean_rel_tol * np.abs(means_np[i:].mean())
            diff_thr = self.diff_abs_tol + self.diff_rel_tol * np.abs(cycle_diffs[i:].mean())

            if mean_std <= mean_thr and diff_std <= diff_thr:
                means_limit_cycle.append(means_np[i])
                diffs_limit_cycle.append(cycle_diffs[i])
            else:
                break
        is_limit_cycle = len(means_limit_cycle) > 0
        return is_limit_cycle, means_limit_cycle, diffs_limit_cycle

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            peaks = s.features.get("peak_indices", None)
            if peaks is None:
                raise ValueError("peak_indices not found, apply PeaksFeature first.")

            cycles = [
                self.split_into_cycles(p, self.q, self.gap_factor) for p in peaks
            ]  # (n_channels, n_cycles, n_peaks_in_cycle), not rectangular

            # Detection needs to be done for each channel separately
            results = [
                self.detect_limit_cycle(ts, ts_cycles)
                for ts, ts_cycles in zip(s.time_series, cycles, strict=True)
            ]
            is_limit_cycle = bool(np.all([r[0] for r in results]))
            if is_limit_cycle:
                min_len = min([len(r[1]) for r in results])  # type: ignore
                means_limit_cycle = np.vstack([r[1][:min_len] for r in results])  # type: ignore
                means_limit_cycle_mean = means_limit_cycle.mean(axis=1).tolist()
                means_limit_cycle = means_limit_cycle.tolist()
                diffs_limit_cycle = np.vstack([r[2][:min_len] for r in results])  # type: ignore
                diffs_limit_cycle_mean = diffs_limit_cycle.mean(axis=1).tolist()
                diffs_limit_cycle = diffs_limit_cycle.tolist()
            else:
                n = s.time_series.shape[0]
                means_limit_cycle = [[None]] * n  # type: ignore
                means_limit_cycle_mean = [None] * n
                diffs_limit_cycle = [[None]] * n  # type: ignore
                diffs_limit_cycle_mean = [None] * n
            s.features["is_limit_cycle"] = is_limit_cycle
            s.features["limit_cycle_means"] = means_limit_cycle
            s.features["limit_cycle_mean"] = means_limit_cycle_mean
            s.features["limit_cycle_diffs"] = diffs_limit_cycle
            s.features["limit_cycle_mean_diff"] = diffs_limit_cycle_mean

        return samples


class ShortenTimeSeriesTransformer:
    """
    Transformer which shortens the time series by `n_steps` time steps for forecasting.
    The last `n_steps` time step values are extracted for forecasting.

    Args:
        n_steps: number of time steps to shorten the time series by (default: 1)
    """

    pre_split: bool = False

    def __init__(self, n_steps: int = 1):
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        self.n_steps = n_steps

    def transform(self, samples: Sequence[TimeSeriesSample]) -> Sequence[TimeSeriesSample]:
        for s in samples:
            s.time_series, forecast_values = np.split(s.time_series, [-self.n_steps], axis=1)
            s.time_points = s.time_points[: -self.n_steps]
            s.config.solver.n_time_points = len(s.time_points)
            s.config.solver.time_span = (s.config.solver.time_span[0], s.time_points[-1])
            s.features["forecast_values"] = forecast_values.tolist()
        return samples
