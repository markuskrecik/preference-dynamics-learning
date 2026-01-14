from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration object for model architecture specification.
    """

    model_type: str = Field(..., description="Type of model architecture")
    model_name: str = Field(..., description="Model identifier")
    in_channels: int = Field(..., gt=0, description="Number of input channels")


class CNN1DConfig(ModelConfig):
    """
    Configuration object for CNN1D model architecture specification.

    Fields:
        model_name: Model identifier
        in_channels: Number of input channels (typically 2 * n_actions)
        filters: Output dimensions of LazyConv1d blocks (length = number of conv blocks)
        kernel_sizes: Kernel sizes for each conv block (same length as filters)
        features: Output dimensions of LazyLinear blocks (length = number of FC blocks)
        dropout: Dropout rate (0.0 to 1.0)
    """

    model_type: str = "cnn1d"
    model_name: str = Field(..., description="Model identifier")
    in_channels: int = Field(..., gt=0, description="Number of input channels")
    filters: Sequence[int] = Field(
        ..., min_length=1, description="Output dimensions of conv blocks"
    )
    kernel_sizes: Sequence[int] = Field(
        ..., min_length=1, description="Kernel sizes for each conv block"
    )
    features: Sequence[int] = Field(..., min_length=1, description="Output dimensions of FC blocks")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate")

    @model_validator(mode="after")
    def validate_filters_kernel_sizes_match(self) -> Self:
        """Validate that filters and kernel_sizes have same length."""
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                f"filters and kernel_sizes must have same length, "
                f"got {len(self.filters)} and {len(self.kernel_sizes)}"
            )
        return self

    @model_validator(mode="after")
    def validate_positive_values(self) -> Self:
        """Validate that all integer values are > 0."""
        if any(f <= 0 for f in self.filters):
            raise ValueError("All filter values must be > 0")
        if any(k <= 0 for k in self.kernel_sizes):
            raise ValueError("All kernel_size values must be > 0")
        if any(f <= 0 for f in self.features):
            raise ValueError("All feature values must be > 0")
        return self


class CNN1DFeatConfig(ModelConfig):
    """
    Configuration object for CNN1D with features model architecture specification.

    Fields:
        model_name: Model identifier
        in_channels: Number of input channels (typically 2 * n_actions)
        filters: Output dimensions of LazyConv1d blocks (length = number of conv blocks)
        kernel_sizes: Kernel sizes for each conv block (same length as filters)
        features: Output dimensions of LazyLinear blocks (length = number of FC blocks)
        dropout: Dropout rate (0.0 to 1.0)
    """

    model_type: str = "cnn1d_feat"
    model_name: str = Field(..., description="Model identifier")
    in_channels: int = Field(..., gt=0, description="Number of input channels")
    filters: Sequence[int] = Field(
        ..., min_length=1, description="Output dimensions of conv blocks"
    )
    kernel_sizes: Sequence[int] = Field(
        ..., min_length=1, description="Kernel sizes for each conv block"
    )
    features: Sequence[int] = Field(..., min_length=1, description="Output dimensions of FC blocks")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate")

    @model_validator(mode="after")
    def validate_filters_kernel_sizes_match(self) -> Self:
        """Validate that filters and kernel_sizes have same length."""
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                f"filters and kernel_sizes must have same length, "
                f"got {len(self.filters)} and {len(self.kernel_sizes)}"
            )
        return self

    @model_validator(mode="after")
    def validate_positive_values(self) -> Self:
        """Validate that all integer values are > 0."""
        if any(f <= 0 for f in self.filters):
            raise ValueError("All filter values must be > 0")
        if any(k <= 0 for k in self.kernel_sizes):
            raise ValueError("All kernel_size values must be > 0")
        if any(f <= 0 for f in self.features):
            raise ValueError("All feature values must be > 0")
        return self


class CNN1DResidualConfig(ModelConfig):
    """
    Configuration object for CNN1D with residual time series and features model architecture specification.

    Fields:
        model_name: Model identifier
        in_channels: Number of input channels (typically 2 * n_actions)
        filters: Output dimensions of LazyConv1d blocks (length = number of conv blocks)
        kernel_sizes: Kernel sizes for each conv block (same length as filters)
        hidden_dims: Output dimensions of LazyLinear blocks (length = number of FC blocks)
        dropout: Dropout rate (0.0 to 1.0)
    """

    model_type: str = "cnn1d_residual"
    model_name: str = Field(..., description="Model identifier")
    in_channels: int = Field(..., gt=0, description="Number of input channels")
    filters: Sequence[int] = Field(
        ..., min_length=1, description="Output dimensions of conv blocks"
    )
    kernel_sizes: Sequence[int] = Field(
        ..., min_length=1, description="Kernel sizes for each conv block"
    )
    hidden_dims: Sequence[int] = Field(
        ..., min_length=1, description="Output dimensions of FC blocks"
    )
    out_dim: int = Field(..., gt=0, description="Output dimension")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate")

    @model_validator(mode="after")
    def validate_filters_kernel_sizes_match(self) -> Self:
        """Validate that filters and kernel_sizes have same length."""
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                f"filters and kernel_sizes must have same length, "
                f"got {len(self.filters)} and {len(self.kernel_sizes)}"
            )
        return self

    @model_validator(mode="after")
    def validate_positive_values(self) -> Self:
        """Validate that all integer values are > 0."""
        if any(f <= 0 for f in self.filters):
            raise ValueError("All filter values must be > 0")
        if any(k <= 0 for k in self.kernel_sizes):
            raise ValueError("All kernel_size values must be > 0")
        if any(f <= 0 for f in self.hidden_dims):
            raise ValueError("All hidden_dim values must be > 0")
        return self
