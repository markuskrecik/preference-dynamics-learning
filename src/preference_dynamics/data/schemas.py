"""
Data schemas
"""

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from preference_dynamics.data.adapters import ParameterTargetAdapter, StateInputAdapter
from preference_dynamics.data.io_handler import JSONHandler


class DataConfig(BaseModel):
    data_dir: str = Field(
        ...,
        description="Root data directory containing `/raw` and `/processed` subdirectories",
    )
    io_handler: Any = Field(
        default_factory=JSONHandler,
        description="IO handler for data loading and saving (default: JSONHandler)",
    )
    transformers: list[Any] = Field(
        default_factory=list,
        description="List of Transformer instances to apply to the data (default: [])",
    )
    input_adapter: Any = Field(
        StateInputAdapter(),
        description="InputAdapter instance for converting samples to model inputs",
    )
    target_adapter: Any = Field(
        ParameterTargetAdapter(),
        description="TargetAdapter instance for converting samples to model targets",
    )
    load_if_exists: bool = Field(
        default=True, description="Whether to load processed data if available (default: True)"
    )
    splits: tuple[float, float, float] = Field(
        default=(0.7, 0.15, 0.15),
        description="Train/val/test split ratios (must sum to 1.0). Default: (0.7, 0.15, 0.15)",
    )
    batch_size: int = Field(
        default=32, ge=1, description="Batch size for data loaders (default: 32)"
    )
    shuffle_train: bool = Field(
        default=True, description="Whether to shuffle training data (default: True)"
    )
    num_workers: int = Field(
        default=-1,
        ge=-1,
        description="Number of worker processes for data loading (-1 = auto-detect, default: -1)",
    )
    pin_memory: bool = Field(
        default=True, description="Whether to pin memory for GPU transfer (default: True)"
    )
    seed: int = Field(
        default=42, ge=0, description="Random seed for deterministic splitting (default: 42)"
    )

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        """Validate that splits sum to 1.0 and are all > 0."""
        if abs(sum(self.splits) - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0, got {sum(self.splits)}")
        if any(s <= 0 for s in self.splits):
            raise ValueError(f"All splits must be > 0, got {self.splits}")
        return self
