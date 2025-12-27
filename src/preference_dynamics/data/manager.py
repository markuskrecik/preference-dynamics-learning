"""
DataManager for unified data management.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Self, get_args

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from preference_dynamics.data.dataset import TimeSeriesDataset
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.schemas import SampleStatistics, TimeSeriesSample

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SplitName = Literal["train", "val", "test"]


class DataManager:
    """
    Unified data management component.

    Handles loading, preprocessing, splitting, and data loader creation
    with support for preprocessed data caching.

    Example:
        ```python
        from preference_dynamics.data import DataManager, DataConfig

        data_config = DataConfig(
            data_dir="data/n1",
            load_if_exists=True,
            splits=(0.7, 0.15, 0.15),
            batch_size=32,
            seed=42
        )
        data_manager = DataManager(config=data_config)

        # Setup data loaders (loads preprocessed data if available)
        data_manager.setup()

        # Access data loaders
        train_loader = data_manager.train_dataloader
        val_loader = data_manager.val_dataloader
        test_loader = data_manager.test_dataloader
        ```
    """

    def __init__(self, config: DataConfig) -> None:
        """
        Initialize data manager.

        Args:
            config: DataConfig with data directory, IO handler, transformers, and loader settings
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.io_handler = config.io_handler
        self._splits: dict[SplitName, list[TimeSeriesSample]] = {}
        self._statistics: dict[SplitName, SampleStatistics] = {}
        self._dataloaders: dict[SplitName, DataLoader] = {}

    def setup(self) -> Self:
        """
        Setup data loaders.

        If processed data exists and load_if_exists=True, loads it.
        Otherwise, loads raw data, applies preprocessing transformations, splits, and saves.

        Raises:
            FileNotFoundError: If raw data not found and processed data not available
        """

        if self.config.load_if_exists:
            suffix = self.io_handler.suffix

            paths_exist = []
            for split_name in get_args(SplitName):
                path = self.processed_dir / f"{split_name}{suffix}"
                exists = path.exists()
                paths_exist.append(exists)
            if all(paths_exist):
                self.load_processed()
                return self

        all_samples = self.load_raw()

        for t in self.config.transformers:
            if t.pre_split:
                all_samples = t.transform(all_samples)

        self._splits = self.train_val_test_split(all_samples, self.config.splits, self.config.seed)

        for t in self.config.transformers:
            if not t.pre_split:
                self._splits = {
                    split_name: t.transform(samples) for split_name, samples in self._splits.items()
                }

        self._create_dataloaders()

        self.save_processed()

        return self

    def train_val_test_split(
        self,
        samples: list[TimeSeriesSample],
        splits: tuple[float, float, float],
        seed: int,
    ) -> dict[SplitName, list[TimeSeriesSample]]:
        """
        Split samples into train/val/test sets deterministically.

        Args:
            samples: Samples to split
            splits: Train/val/test ratios (must sum to 1.0)
            seed: Random seed for deterministic splitting

        Returns:
            Dictionary with split names as keys and lists of samples as values

        Raises:
            ValueError: If splits don't sum to 1.0 or any value <= 0
        """
        train_ratio, val_ratio, test_ratio = splits

        if len(samples) == 0:
            return {
                "train": [],
                "val": [],
                "test": [],
            }

        # First split: train vs (val+test)
        train_samples, temp_samples = train_test_split(
            samples,
            train_size=train_ratio,
            random_state=seed,
        )

        # Second split: val vs test
        temp_total_ratio = val_ratio + test_ratio
        test_size = test_ratio / temp_total_ratio if temp_total_ratio > 0 else 0.0
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=test_size,
            random_state=seed + 1,
        )

        return {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

    def save_raw(self, samples: list[TimeSeriesSample], filename: str = "samples") -> None:
        """Save raw data to raw directory."""
        logger.info(f"Saving {len(samples)} raw samples to {self.raw_dir}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.io_handler.save(samples, self.raw_dir / filename)

    def load_raw(self) -> list[TimeSeriesSample]:
        """Load raw data from raw directory."""
        logger.info(f"Loading raw data from {self.raw_dir}")
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_dir}. "
                "Either provide raw data or preprocessed data."
            )
        suffix = self.io_handler.suffix
        raw_files = list(self.raw_dir.glob(f"*{suffix}"))
        if not raw_files:
            raise FileNotFoundError(f"No {suffix}-files found in {self.raw_dir}.")

        all_samples: list[TimeSeriesSample] = []
        for raw_file in raw_files:
            samples = self.io_handler.load(raw_file)
            all_samples.extend(samples)
        return all_samples

    def save_processed(self) -> None:
        """
        Save preprocessed data splits to files using configured IO handler.

        Saves train/val/test splits to data_dir/processed/ using the configured
        IO handler (e.g., JSONHandler).

        Raises:
            RuntimeError: If no processed data to save (setup not called or no data)
        """
        if not self._splits:
            raise RuntimeError("No data to save. Call setup() first.")

        logger.info(f"Saving {len(self._splits)} splits to {self.processed_dir}")
        suffix = self.io_handler.suffix
        for split_name, samples in self._splits.items():
            path = self.processed_dir / f"{split_name}{suffix}"
            self.io_handler.save(samples, path)
            logger.debug(f"Saved {len(samples)} samples to {path}")

    def _load_processed_splits(self) -> None:
        """Load splits from processed directory."""
        suffix = self.io_handler.suffix
        for split_name in get_args(SplitName):
            path = self.processed_dir / f"{split_name}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Processed {split_name} data not found: {path}")
            samples = self.io_handler.load(path)
            self._splits[split_name] = samples

    def load_processed(self) -> None:
        """
        Load preprocessed data from files using configured IO handler.

        Loads train/val/test splits from data_dir/processed/ and creates data loaders.

        Raises:
            FileNotFoundError: If processed data not found
            ValueError: If processed data is corrupted or incompatible
        """
        logger.info(f"Loading preprocessed data from {self.processed_dir}")
        self._load_processed_splits()
        self._create_dataloaders()
        logger.info(
            f"Loaded {sum(len(samples) for samples in self._splits.values())} total samples"
        )

    def _create_dataloaders(self) -> None:
        """Create data loaders from splits."""
        if not self._splits:
            raise RuntimeError("No splits available. Call setup() or load_processed() first.")

        if self.config.num_workers == -1:
            num_workers = os.cpu_count() or 0
        else:
            num_workers = self.config.num_workers

        for split_name, samples in self._splits.items():
            dataset = TimeSeriesDataset(
                samples,
                input_adapter=self.config.input_adapter,
                target_adapter=self.config.target_adapter,
            )
            self._dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_train if split_name == "train" else False,
                num_workers=num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=True,
            )

    @property
    def splits(self) -> dict[SplitName, list[TimeSeriesSample]]:
        """Get splits."""
        return self._splits

    @property
    def train_dataloader(self) -> DataLoader:
        """
        Get training data loader.

        Returns:
            Training data loader

        Raises:
            RuntimeError: If setup() not called
        """
        if self._dataloaders.get("train") is None:
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self._dataloaders["train"]

    @property
    def val_dataloader(self) -> DataLoader | None:
        """
        Get validation data loader.

        Returns:
            Validation data loader or None if no validation split

        Raises:
            RuntimeError: If setup() not called
        """
        if self._dataloaders.get("train") is None:
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self._dataloaders.get("val")

    @property
    def test_dataloader(self) -> DataLoader | None:
        """
        Get test data loader.

        Returns:
            Test data loader or None if no test split

        Raises:
            RuntimeError: If setup() not called
        """
        if self._dataloaders.get("train") is None:
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self._dataloaders.get("test")

    @property
    def n_inputs(self) -> int:
        """Get number of inputs."""
        sample = self.splits["train"][0]
        return self.config.input_adapter.n_inputs(sample)  # type: ignore

    @property
    def n_targets(self) -> int:
        """Get number of targets."""
        sample = self.splits["train"][0]
        return self.config.target_adapter.n_targets(sample)  # type: ignore
