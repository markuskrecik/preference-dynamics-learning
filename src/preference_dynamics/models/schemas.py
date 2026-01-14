from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration object for model architecture specification.
    """

    model_type: str = Field(..., description="Type of model architecture")
    model_name: str = Field(..., description="Model identifier")
    # in_channels: int = Field(..., gt=0, description="Number of input channels")


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


class TimeSeriesEncoderConfig(ModelConfig):
    """
    Configuration for time-series encoder that maps trajectories to parameter/IC estimates.

    Fields:
        model_type: Model type identifier (default: "cnn1d")
        in_dims: Input dimensions as (channels, length) tuple
        filters: Output dimensions of LazyConv1d blocks (length = number of conv blocks)
        kernel_sizes: Kernel sizes for each conv block (same length as filters)
        hidden_dims: Output dimensions of LazyLinear blocks (length = number of FC blocks)
        out_dims: Output dimensions (total params + IC size)
        dropout: Dropout rate (0.0 to 1.0)
    """

    model_type: str = "cnn1d"
    model_name: str = Field(..., description="Model identifier")
    in_dims: tuple[int, int] = Field(
        ..., description="Input dimensions as (channels, length) tuple"
    )
    filters: Sequence[int] = Field(
        ..., min_length=1, description="Output dimensions of conv blocks"
    )
    kernel_sizes: Sequence[int] = Field(
        ..., min_length=1, description="Kernel sizes for each conv block"
    )
    hidden_dims: Sequence[int] = Field(
        ..., min_length=1, description="Output dimensions of FC blocks"
    )
    out_dims: int = Field(..., gt=0, description="Output dimensions (total params + IC size)")
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
        if self.in_dims[0] <= 0 or self.in_dims[1] <= 0:
            raise ValueError("in_dims values must be > 0")
        if any(f <= 0 for f in self.filters):
            raise ValueError("All filter values must be > 0")
        if any(k <= 0 for k in self.kernel_sizes):
            raise ValueError("All kernel_size values must be > 0")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("All hidden_dims values must be > 0")
        return self


class SurrogateConfig(ModelConfig):
    """
    Configuration for trajectory surrogate that maps (t, θ, x₀) → x(t).

    Fields:
        model_type: Model type identifier (default: "surrogate")
        in_dims: Input dimensions as (channels, length) tuple
        hidden_dims: Hidden layer dimensions for MLP
        out_dims: Output dimensions as (channels, length) tuple
        dropout: Dropout rate (0.0 to 1.0)
    """

    model_type: str = "surrogate"
    model_name: str = Field(..., description="Model identifier")
    in_dims: Sequence[int] = Field(..., description="Input dimensions as (channels, length) tuple")
    hidden_dims: Sequence[int] = Field(
        ..., min_length=1, description="Hidden layer dimensions for MLP"
    )
    out_dims: Sequence[int] = Field(
        ..., description="Output dimensions as (channels, length) tuple"
    )
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")

    @model_validator(mode="after")
    def validate_positive_values(self) -> Self:
        """Validate that all integer values are > 0."""
        if any(d <= 0 for d in self.in_dims):
            raise ValueError("All in_dims values must be > 0")
        if any(d <= 0 for d in self.out_dims):
            raise ValueError("out_dims values must be > 0")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("All hidden_dims values must be > 0")
        return self


class InversePINNConfig(ModelConfig):
    """
    Configuration for amortized inverse PINN model composition.

    The model consists of:
        - encoder: maps trajectory → (θ̂, x̂₀)
        - surrogate: maps (t, θ̂, x̂₀) → x̂(t)
        - parameter_transform: enforces constraints on θ̂

    Fields:
        model_type: Model type identifier (default: "inverse_pinn")
        model_name: Model identifier
        encoder: Configuration for time-series encoder
        surrogate: Configuration for conditional trajectory surrogate
    """

    model_type: str = "inverse_pinn"
    model_name: str = Field(..., description="Model identifier")
    encoder: TimeSeriesEncoderConfig = Field(..., description="Time-series encoder configuration")
    surrogate: SurrogateConfig = Field(..., description="Trajectory surrogate configuration")
