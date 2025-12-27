"""
PyTorch Dataset classes for preference dynamics time series.
"""

import mlflow
import torch
from torch.utils.data import Dataset

from preference_dynamics.data.adapters import InputAdapter, TargetAdapter
from preference_dynamics.schemas import TimeSeriesSample
from preference_dynamics.utils import to_cpu_numpy


class TimeSeriesDataset(Dataset):  # type: ignore
    """
    PyTorch Dataset for preference dynamics time series.
    """

    def __init__(
        self,
        samples: list[TimeSeriesSample],
        input_adapter: InputAdapter,
        target_adapter: TargetAdapter,
    ) -> None:
        """
        Initialize dataset from list of samples.

        Args:
            samples: List of TimeSeriesSample objects (must all have same n_actions)
            input_adapter: InputAdapter to use for converting samples to model inputs
            target_adapter: TargetAdapter to use for converting samples to model targets

        Raises:
            ValueError: If samples list is empty or has inconsistent n_actions
        """
        if len(samples) == 0:
            raise ValueError("Cannot create dataset from empty samples list")

        n_actions = samples[0].n_actions
        for sample in samples:
            if sample.n_actions != n_actions:
                raise ValueError(
                    f"Sample {sample.sample_id} has n_actions={sample.n_actions}, "
                    f"expected {n_actions}"
                )

        self.n_actions = n_actions
        self.samples = samples
        self.input_adapter = input_adapter
        self.target_adapter = target_adapter

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - inputs: dict[str, torch.Tensor]
                - targets: dict[str, torch.Tensor]
        """
        sample = self.samples[idx]

        return {
            "inputs": self.input_adapter.get_inputs(sample),
            "targets": self.target_adapter.get_targets(sample),
        }

    @property
    def signature(self) -> mlflow.models.ModelSignature:
        """
        Get MLflow signature for a sample.
        """

        sample = self.__getitem__(0)
        inputs = to_cpu_numpy(sample["inputs"])
        targets = to_cpu_numpy(sample["targets"])
        return mlflow.models.infer_signature(inputs, targets)
