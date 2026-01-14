from pathlib import Path

import pytest

from preference_dynamics.data.io_handler import JSONHandler
from preference_dynamics.schemas import TimeSeriesSample


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def model_dir() -> Path:
    return Path(__file__).parent.parent / "model"


@pytest.fixture(scope="module")
def sample_data(
    data_dir: Path,
) -> tuple[list[TimeSeriesSample], list[TimeSeriesSample], list[TimeSeriesSample]]:
    io_handler = JSONHandler()
    processed_dir = data_dir / "processed"

    train_samples = io_handler.load(processed_dir / "train.json")
    val_samples = io_handler.load(processed_dir / "val.json")
    test_samples = io_handler.load(processed_dir / "test.json")

    return train_samples, val_samples, test_samples
