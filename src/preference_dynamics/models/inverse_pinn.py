"""
Amortized inverse PINN model for ODE parameter and initial condition inference.

The model consists of:
    - encoder: maps trajectory → (θ̂, x̂₀)
    - surrogate: maps (t, θ̂, x̂₀) → x̂(t)
    - parameter_transform: enforces constraints on θ̂
"""

import torch
from torch import nn

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.models.schemas import (
    InversePINNConfig,
    SurrogateConfig,
    TimeSeriesEncoderConfig,
)
from preference_dynamics.models.utils import get_jacobian_diagonal


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
        super().__init__(config)
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
            x: Observed time series of shape (..., in_channels, T)

        Returns:
            params_unconstrained: Unconstrained parameter estimates of shape (B, n_params)
            ic: Initial condition estimates of shape (B, n_ic)
        """
        # if x.ndim == 2:
        #     x = x.unsqueeze(0)
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
        super().__init__(config)
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
            params: Parameters of shape (..., n_params)
            ic: Initial conditions of shape (..., n_ic)
            t: Time points of shape (..., T)

        Returns:
            Dictionary with:
                latent_x: Latent state [v̂(t), m̂(t)] of shape (..., 2n, T)
                d_latent_x: Time derivative d(latent_x)/dt of shape (..., 2n, T)
        """
        if not t.ndim == params.ndim == ic.ndim:
            raise ValueError("Inputs must have same dimensions")

        is_batched = t.ndim == 2
        T = t.shape[-1]
        N = ic.shape[-1]  # number of channels
        t.requires_grad_(True)

        def f(t: torch.Tensor, params: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
            x = torch.cat([t, params, ic], dim=-1)  # (..., n + 2n + 2n² + T)
            y = self.mlp(x)
            y = y.view(-1, N, T) if y.is_contiguous() else y.reshape(-1, N, T)  # (B or 1, N, T)
            if not is_batched:
                y.squeeze_(0)
            return y

        latent_x = f(t, params, ic)
        # d_latent_x = get_jacobian_diagonal_autograd(latent_x, t)
        d_latent_x = get_jacobian_diagonal(f, t, params, ic)
        # Remove spurious dim:
        if d_latent_x.ndim == 4 and d_latent_x.shape[1] == 1:
            d_latent_x.squeeze_(1)

        return {
            "latent_x": latent_x,
            "d_latent_x": d_latent_x,
        }


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
        super().__init__(config)

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
                - d_latent_x: Time derivative of latent state of shape (B, 2n, T)
                - t: Time points
        """
        params, ic = self.encoder(x)
        surrogate_output = self.surrogate(params, ic, t)

        return {
            "params": params,
            "ic": ic,
            "latent_x": surrogate_output["latent_x"],
            "d_latent_x": surrogate_output["d_latent_x"],
        }
