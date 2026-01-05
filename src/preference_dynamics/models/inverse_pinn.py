"""
Amortized inverse PINN model for ODE parameter and initial condition inference.

The model consists of:
    - encoder: maps trajectory → (θ̂, x̂₀)
    - surrogate: maps (t, θ̂, x̂₀) → x̂(t)
    - parameter_transform: enforces constraints on θ̂
"""

from pathlib import Path

import torch
from torch import nn

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.models.schemas import (
    InversePINNConfig,
    SurrogateConfig,
    TimeSeriesEncoderConfig,
)


class TimeSeriesEncoder(PredictorModel):
    """
    Time-series encoder that maps observed trajectories to parameter and IC estimates.

    Architecture:
        - 1D convolutional blocks for temporal feature extraction
        - Global pooling to handle variable-length sequences
        - Fully connected layers to output (θ̂, x̂₀)
    """

    def __init__(self, config: TimeSeriesEncoderConfig) -> None:
        """
        Args:
            config: TimeSeriesEncoderConfig
        """
        super().__init__()
        self._config = config
        # in_channels = config.in_dims[0]
        # self.n_actions = in_channels // 2
        # self.n_params = 2 * self.n_actions + 2 * self.n_actions**2
        # self.n_ic = 2 * self.n_actions
        self.n_output = config.out_dims

        # if self.n_output != self.n_params + self.n_ic:
        #     raise ValueError(
        #         f"out_dims ({self.n_output}) must equal n_params + n_ic ({self.n_params + self.n_ic})"
        #     )

        self.conv_blocks = nn.ModuleList()
        for out_channels, kernel_size in zip(config.filters, config.kernel_sizes, strict=True):
            conv_block = nn.Sequential(
                nn.LazyConv1d(
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
            self.conv_blocks.append(conv_block)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(config.dropout),
        )

        self.fc_layers = nn.ModuleList()
        for feature_dim in config.hidden_dims:
            fc_layer = nn.Sequential(
                nn.LazyLinear(feature_dim),
                nn.ReLU(inplace=True),
            )
            self.fc_layers.append(fc_layer)

        self.output_layer = nn.LazyLinear(self.n_output)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory to parameter and IC estimates.

        Args:
            x: Observed time series of shape (B, in_channels, T)

        Returns:
            params_unconstrained: Unconstrained parameter estimates of shape (B, n_params)
            ic: Initial condition estimates of shape (B, n_ic)
        """
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        output = self.output_layer(x)
        params_unconstrained = output[:, : self.n_params]
        ic = output[:, self.n_params :]

        return params_unconstrained, ic

    @property
    def config(self) -> TimeSeriesEncoderConfig:
        """Configuration for the inverse PINN model."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Return number of trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Surrogate(PredictorModel):
    """
    Trajectory surrogate that maps (θ, x₀, t) → x̂(t).

    Architecture:
        - MLP that takes concatenated [θ, x₀, t] as input
        - Outputs latent state x̂(t) = [v̂(t), m̂(t)]
    """

    def __init__(self, config: SurrogateConfig) -> None:
        """
        Args:
            config: SurrogateConfig
        """
        super().__init__()
        self._config = config
        # in_dim = sum(config.in_dims)
        out_dim = config.out_dims[-1] * config.out_dims[-2]

        layers = []
        # prev_dim = in_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            # prev_dim = hidden_dim

        layers.append(nn.LazyLinear(out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        t: torch.Tensor,
        params: torch.Tensor,
        ic: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict latent state at time t given parameters and initial conditions.

        Args:
            t: Time points of shape (..., T)
            params: Parameters of shape (..., n_params)
            ic: Initial conditions of shape (..., n_ic)

        Returns:
            latent_state: Latent state [v̂(t), m̂(t)] of shape (..., 2n, T)
        """

        # print(t.shape, params.shape, ic.shape)
        t_shape = tuple(t.shape)
        if len(t_shape) == 1:
            t_shape = (1, *t_shape)
        t.requires_grad_(True)
        # params_expanded = params.unsqueeze(-2).expand(*t_shape, -1)
        # ic_expanded = ic.unsqueeze(-2).expand(*t_shape, -1)
        # t_expanded = t.unsqueeze(-1)
        # print(params_expanded.shape, ic_expanded.shape, t_expanded.shape)
        # x = torch.cat([params_expanded, ic_expanded, t_expanded], dim=-1)
        x = torch.cat([params, ic, t], dim=-1)

        x = self.mlp(x)
        x = x.view(*t_shape, -1)  # (B, T, 2n)

        return {"latent_x": x.transpose(1, 2), "t": t}

    @property
    def config(self) -> SurrogateConfig:
        """Configuration for the inverse PINN model."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Return number of trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InversePINNPredictor(PredictorModel):
    """
    Composed inverse PINN model combining encoder, surrogate, and parameter transforms.

    The model:
        1. Encodes observed trajectory → (θ̂, x̂₀)
        2. Applies parameter constraint transform to θ̂
        3. Uses surrogate to predict latent state x̂(t) from (t, θ̂, x̂₀)
        4. Derives observables (û, â) from x̂(t)
    """

    def __init__(self, config: InversePINNConfig) -> None:
        """
        Args:
            config: Inverse PINN configuration
        """
        super().__init__()
        self._config = config

        self.encoder = TimeSeriesEncoder(config.encoder)
        self.surrogate = Surrogate(config.surrogate)

        in_channels = config.encoder.in_dims[0]
        self.n_actions = in_channels // 2

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through inverse PINN.

        Args:
            x: Observed time series of shape (B, in_channels, T)
            t: Time points of shape (B, T) or (B, T) with requires_grad=True

        Returns:
            Dictionary with:
                - params: Parameter estimates of shape (B, 2n(n+1))
                - ic: Initial condition estimates of shape (B, 2n)
                - latent_x: Latent state trajectory of shape (B, 2n, T)
        """
        params, ic = self.encoder(x)
        latent_state = self.surrogate(t, params, ic)

        return {"params": params, "ic": ic, "latent_x": latent_state, "t": t}

    def save(self, path: str) -> None:
        """Save model state and config to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else None,
            "model_type": self.config.model_type,
        }

        torch.save(checkpoint, save_path)

    def load(self, path: str) -> None:
        """Load model state from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()

    @property
    def config(self) -> InversePINNConfig:
        """Configuration for the inverse PINN model."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Return number of trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
