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
from preference_dynamics.models.schemas import InversePINNConfig
from preference_dynamics.schemas import ICVector, ParameterVector
from preference_dynamics.solver.torch_equations import compute_observables_torch
from preference_dynamics.utils import get_diagonal_indices


class TimeSeriesEncoder(nn.Module):  # type: ignore
    """
    Time-series encoder that maps observed trajectories to parameter and IC estimates.

    Architecture:
        - 1D convolutional blocks for temporal feature extraction
        - Global pooling to handle variable-length sequences
        - Fully connected layers to output (θ̂, x̂₀)
    """

    def __init__(
        self,
        in_channels: int,
        filters: list[int],
        kernel_sizes: list[int],
        features: list[int],
        dropout: float,
        n_actions: int,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels (typically 2 * n_actions)
            filters: Output dimensions of conv blocks
            kernel_sizes: Kernel sizes for each conv block
            features: Output dimensions of FC blocks
            dropout: Dropout rate
            n_actions: Number of actions (n)
        """
        super().__init__()
        self.n_actions = n_actions
        self.n_params = 2 * n_actions + 2 * n_actions**2
        self.n_ic = 2 * n_actions
        self.n_output = self.n_params + self.n_ic

        self.conv_blocks = nn.ModuleList()
        for out_channels, kernel_size in zip(filters, kernel_sizes, strict=True):
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
            nn.Dropout(dropout),
        )

        self.fc_layers = nn.ModuleList()
        for feature_dim in features:
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


class Surrogate(nn.Module):  # type: ignore
    """
    Trajectory surrogate that maps (t, θ̂, x̂₀) → x̂(t).

    Architecture:
        - MLP that takes concatenated [t, θ̂, x̂₀] as input
        - Outputs latent state x̂(t) = [v̂(t), m̂(t)]
    """

    def __init__(
        self,
        n_params: int,
        n_ic: int,
        hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            n_params: Number of parameters (2n + 2n²)
            n_ic: Number of initial conditions (2n)
            hidden_dims: Hidden layer dimensions for MLP
            activation: Activation function name
            dropout: Dropout rate
        """
        super().__init__()
        self.n_params = n_params
        self.n_ic = n_ic
        self.n_state = n_ic
        input_dim = 1 + n_params + n_ic

        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_map[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.n_state))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        t: torch.Tensor,
        params: torch.Tensor,
        ic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict latent state at time t given parameters and initial conditions.

        Args:
            t: Time points of shape (B, T) or (T,)
            params: Parameter estimates of shape (B, n_params)
            ic: Initial condition estimates of shape (B, n_ic)

        Returns:
            state: Latent state [v̂(t), m̂(t)] of shape (B, T, n_state) or (T, n_state)
        """
        # if t.ndim == 1:
        #     t = t.unsqueeze(0)
        #     batch_mode = False
        # else:
        #     batch_mode = True

        # B, T = t.shape
        # params_expanded = params.unsqueeze(1).expand(B, T, -1)
        # ic_expanded = ic.unsqueeze(1).expand(B, T, -1)
        # t_expanded = t.unsqueeze(-1)

        # x = torch.cat([t_expanded, params_expanded, ic_expanded], dim=-1)
        x = torch.cat([t, params, ic], dim=-1)
        state = self.mlp(x)

        # if not batch_mode:
        #     state = state.squeeze(0)

        return state


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

        n_actions = self._infer_n_actions(config.in_channels)

        self.encoder = TimeSeriesEncoder(
            in_channels=config.in_channels,
            filters=list(config.encoder.filters),
            kernel_sizes=list(config.encoder.kernel_sizes),
            features=list(config.encoder.features),
            dropout=config.encoder.dropout,
            n_actions=n_actions,
        )

        n_params = 2 * n_actions + 2 * n_actions**2
        n_ic = 2 * n_actions

        self.surrogate = Surrogate(
            n_params=n_params,
            n_ic=n_ic,
            hidden_dims=list(config.surrogate.hidden_dims),
            activation=config.surrogate.activation,
            dropout=config.surrogate.dropout,
        )

        self.parameter_transform_type = config.parameter_transform
        self.n_actions = n_actions

    # TODO: remove
    def _infer_n_actions(self, in_channels: int) -> int:
        """Infer n_actions from in_channels (assumes in_channels = 2 * n_actions)."""
        if in_channels % 2 != 0:
            raise ValueError(f"in_channels must be even, got {in_channels}")
        return in_channels // 2

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_intermediates: bool = False,  # TODO: remove
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through inverse PINN.

        Args:
            x: Observed time series of shape (B, in_channels, T)
            t: Time points of shape (T,) or (B, T)
            return_intermediates: Whether to return intermediate outputs

        Returns:
            If return_intermediates=False:
                observables: Predicted observables [û, â] of shape (B, in_channels, T)
            If return_intermediates=True:
                Dictionary with:
                    - observables: Predicted observables
                    - params_unconstrained: Unconstrained parameter estimates
                    - params_constrained: Constrained parameter estimates
                    - ic: Initial condition estimates
                    - state: Latent state trajectory
        """
        params_unconstrained, ic = self.encoder(x)

        params_constrained = self._apply_parameter_transform(
            params_unconstrained, self.parameter_transform_type
        )

        # TODO: remove?
        t_batch = t.unsqueeze(0).expand(x.shape[0], -1) if t.ndim == 1 else t

        state = self.surrogate(t_batch, params_constrained, ic)

        mu = params_constrained[:, self.n_actions : 2 * self.n_actions]
        mu_expanded = mu.unsqueeze(1).expand(-1, state.shape[1], -1)

        observables = compute_observables_torch(
            state=state,
            mu=mu_expanded,
            n_actions=self.n_actions,
            smooth=False,
        )

        if return_intermediates:
            return {
                "observables": observables,
                "params_unconstrained": params_unconstrained,
                "params_constrained": params_constrained,
                "ic": ic,
                "state": state,
            }
        return observables

    # TODO: move to encoder
    def _apply_parameter_transform(
        self,
        params_unconstrained: torch.Tensor,
        transform_type: str,
    ) -> torch.Tensor:
        """
        Apply parameter constraint transform.

        Args:
            params_unconstrained: Unconstrained parameters of shape (B, n_params)
            transform_type: Transform type ("softplus", "exp", etc.)

        Returns:
            params_constrained: Constrained parameters of shape (B, n_params)
        """
        params_constrained = params_unconstrained.clone()
        diagonal_indices = get_diagonal_indices(self.n_actions)
        for idx in diagonal_indices:
            if transform_type == "softplus":
                params_constrained[:, idx] = torch.nn.functional.softplus(
                    params_unconstrained[:, idx]
                )
            elif transform_type == "exp":
                params_constrained[:, idx] = torch.exp(params_unconstrained[:, idx]) + 1e-8
            return params_constrained
        raise ValueError(f"Unknown transform type: {transform_type}")

    # TODO: remove
    def predict_parameters_and_ic(
        self,
        x: torch.Tensor,
    ) -> tuple[ParameterVector, ICVector]:
        """
        Predict parameters and initial conditions from trajectory.

        Args:
            x: Observed time series of shape (in_channels, T) or (B, in_channels, T)

        Returns:
            params: Parameter vector estimate
            ic: Initial condition vector estimate
        """
        self.eval()
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(0)

            params_unconstrained, ic = self.encoder(x)
            params_constrained = self._apply_parameter_transform(
                params_unconstrained, self.parameter_transform_type
            )

            params_np = params_constrained[0].cpu().numpy()
            ic_np = ic[0].cpu().numpy()

            params = ParameterVector(values=params_np, n_actions=self.n_actions)
            ic = ICVector(values=ic_np, n_actions=self.n_actions)

        return params, ic

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
