"""
PyTorch Dataset classes for preference dynamics time series.
"""

import torch
from torch.utils.data import Dataset

from preference_dynamics.schemas import TimeSeriesSample


class TimeSeriesDataset(Dataset):  # type: ignore
    """
    PyTorch Dataset for preference dynamics time series.
    """

    def __init__(self, samples: list[TimeSeriesSample]) -> None:
        """
        Initialize dataset from list of samples.

        Args:
            samples: List of TimeSeriesSample objects (must all have same n_actions)

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

        self.samples = samples
        self.n_actions = n_actions

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - data: torch.Tensor of shape (2n, T) - variable length time series
                - target: torch.Tensor of shape (2n + 2n²,) - ground truth parameters
                - sequence_length: int - original sequence length (before padding)
                - sample_id: str - unique identifier
        """
        sample = self.samples[idx]

        time_series = torch.from_numpy(sample.time_series.copy()).float()
        parameters = torch.from_numpy(sample.parameters.values.copy()).float()
        sequence_length = sample.sequence_length
        sample_id = sample.sample_id

        return {
            "data": time_series,
            "target": parameters,
            "sequence_length": sequence_length,
            "sample_id": sample_id,
        }
