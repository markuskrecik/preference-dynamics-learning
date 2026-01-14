"""
Unit tests for DataManager class.
"""

import os
from pathlib import Path

import pytest

from preference_dynamics.data.adapters import ParameterTargetAdapter, StateInputAdapter
from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.schemas import TimeSeriesSample


def _write_raw_samples(data_dir: Path, samples: list[TimeSeriesSample], config: DataConfig) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config.io_handler.save(samples, raw_dir / "train")


def test_invalid_split_ratios_raise() -> None:
    with pytest.raises(ValueError):
        DataConfig(
            data_dir="data",
            splits=(0.6, 0.3, 0.3),
            input_adapter=StateInputAdapter(),
            target_adapter=ParameterTargetAdapter(),
        )


def test_train_val_split_empty(tmp_path: Path, random_time_series_n3: TimeSeriesSample) -> None:
    config = DataConfig(
        data_dir=str(tmp_path),
        load_if_exists=False,
        input_adapter=StateInputAdapter(),
        target_adapter=ParameterTargetAdapter(),
    )
    manager = DataManager(config=config)

    empty = manager.train_val_test_split([], (0.7, 0.2, 0.1), seed=0)

    assert empty == {"train": [], "val": [], "test": []}


@pytest.mark.parametrize(
    ("splits", "expected_counts"),
    [((0.7, 0.2, 0.1), (7, 2, 1)), ((0.6, 0.2, 0.2), (6, 2, 2))],
)
def test_train_val_test_split_respects_ratios(
    tmp_path: Path,
    random_time_series_n3: TimeSeriesSample,
    splits: tuple[float, float, float],
    expected_counts: tuple[int, int, int],
) -> None:
    config = DataConfig(
        data_dir=str(tmp_path),
        load_if_exists=False,
        splits=splits,
        input_adapter=StateInputAdapter(),
        target_adapter=ParameterTargetAdapter(),
    )
    manager = DataManager(config=config)
    samples = [random_time_series_n3.model_copy(update={"sample_id": f"s{i}"}) for i in range(10)]

    result = manager.train_val_test_split(samples, splits, seed=123)

    assert (len(result["train"]), len(result["val"]), len(result["test"])) == expected_counts


# TODO: implement
# def test_split_is_deterministic(
#         self, tmp_path: Path, random_time_series_n3: TimeSeriesSample
#     ) -> None:
#         """Test that train_val_test_split() is deterministic with same seed."""


def test_setup_load_if_exists_missing_processed_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    config = DataConfig(
        data_dir=str(data_dir),
        load_if_exists=True,
        input_adapter=StateInputAdapter(),
        target_adapter=ParameterTargetAdapter(),
    )
    manager = DataManager(config=config)

    with pytest.raises(FileNotFoundError):
        manager.setup()


def test_load_raw_missing_files_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    manager = DataManager(
        config=DataConfig(
            data_dir=str(data_dir),
            load_if_exists=False,
            input_adapter=StateInputAdapter(),
            target_adapter=ParameterTargetAdapter(),
        )
    )

    with pytest.raises(FileNotFoundError, match="No .*files found"):
        manager.load_raw()


def test_setup_uses_cpu_count_when_num_workers_neg1(
    data_dir_with_raw_and_processed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu_count = 6
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    config = DataConfig(
        data_dir=str(data_dir_with_raw_and_processed),
        load_if_exists=True,
        num_workers=-1,
        batch_size=2,
        input_adapter=StateInputAdapter(),
        target_adapter=ParameterTargetAdapter(),
    )
    manager = DataManager(config=config)

    manager.setup()

    assert manager.train_dataloader.num_workers == cpu_count
    assert manager.val_dataloader.num_workers == cpu_count
    assert manager.test_dataloader.num_workers == cpu_count


def test_save_processed_writes_all_splits(
    tmp_path: Path, random_time_series_n3: TimeSeriesSample
) -> None:
    data_dir = tmp_path / "data"
    config = DataConfig(
        data_dir=str(data_dir),
        load_if_exists=False,
        batch_size=2,
        input_adapter=StateInputAdapter(),
        target_adapter=ParameterTargetAdapter(),
    )
    samples = [random_time_series_n3.model_copy(update={"sample_id": f"s{i}"}) for i in range(12)]
    _write_raw_samples(data_dir, samples, config)
    manager = DataManager(config=config)

    manager.setup()

    suffix = config.io_handler.suffix
    processed_dir = data_dir / "processed"
    assert (processed_dir / f"train{suffix}").exists()
    assert (processed_dir / f"val{suffix}").exists()
    assert (processed_dir / f"test{suffix}").exists()


def test_load_processed_populates_dataloaders(
    data_dir_with_raw_and_processed: Path,
) -> None:
    config = DataConfig(
        data_dir=str(data_dir_with_raw_and_processed),
        load_if_exists=True,
        input_adapter=StateInputAdapter(),
        target_adapter=ParameterTargetAdapter(),
    )
    manager = DataManager(config=config)

    manager.load_processed()

    assert manager.train_dataloader is not None
    assert manager.val_dataloader is not None
    assert manager.test_dataloader is not None


# TODO: implement: setup and load_processed roundtrip gives same data
