"""
Unit tests for IO handlers.
"""

from pathlib import Path

import numpy as np
import pytest

from preference_dynamics.data.io_handler import IOHandler, JSONHandler, PickleHandler
from preference_dynamics.schemas import TimeSeriesSample


@pytest.mark.parametrize("handler", [JSONHandler(), PickleHandler()])
def test_handler_roundtrip_and_suffix(
    handler: IOHandler, tmp_path: Path, random_time_series_n3: TimeSeriesSample
) -> None:
    path = tmp_path / "sample_path"
    samples = [random_time_series_n3.model_copy(update={"sample_id": f"s{i}"}) for i in range(2)]

    handler.save(samples, path)
    saved_path = path.with_suffix(handler.suffix)
    assert saved_path.exists()

    loaded = handler.load(path)
    assert len(loaded) == len(samples)
    for loaded_sample, original in zip(loaded, samples, strict=True):
        np.testing.assert_array_equal(loaded_sample.time_series, original.time_series)
        np.testing.assert_array_equal(loaded_sample.time_points, original.time_points)
        assert loaded_sample.metadata == original.metadata


@pytest.mark.parametrize("handler", [JSONHandler(), PickleHandler()])
def test_handler_load_missing_raises(handler: IOHandler, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        handler.load(tmp_path / "missing")
