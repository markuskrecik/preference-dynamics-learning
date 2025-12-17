"""
PyTorch Dataset classes for preference dynamics time series.
"""

import mlflow
import torch
from torch.utils.data import Dataset

from preference_dynamics.data.adapters import ModelAdapter
from preference_dynamics.schemas import TimeSeriesSample


class TimeSeriesDataset(Dataset):  # type: ignore
    """
    PyTorch Dataset for preference dynamics time series.
    """

    def __init__(self, samples: list[TimeSeriesSample], adapter: ModelAdapter) -> None:
        """
        Initialize dataset from list of samples.

        Args:
            samples: List of TimeSeriesSample objects (must all have same n_actions)
            adapter: ModelAdapter to use for converting samples to model inputs and targets

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
        self.adapter = adapter

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - input: dict[str, torch.Tensor] - model input
                - target: torch.Tensor - model target
        """
        sample = self.samples[idx]

        return self.adapter(sample)

    @property
    def signature(self) -> mlflow.models.ModelSignature:
        """
        Get MLflow signature for a sample.
        """

        sample = self.__getitem__(0)
        inputs = {k: v.cpu().numpy() for k, v in sample["inputs"].items()}
        target = sample["target"].cpu().numpy()  # type: ignore
        return mlflow.models.infer_signature(inputs, target)
