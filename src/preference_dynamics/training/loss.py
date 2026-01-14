"""
Loss computation utilities.
"""

import torch
from torch import nn

from preference_dynamics.schemas import PINNLossConfig
from preference_dynamics.solver.torch_equations import compute_observables_torch
from preference_dynamics.training.logging import log_PINN_loss


class PINNLoss(nn.Module):  # type: ignore
    """
    Physics-Informed Neural Network loss for inverse PINN training.

    Computes:
        - Data loss: MSE between predicted and target state
        - Physics loss: MSE of ODE residual (d(latent state)/dt - RHS)
        - Supervised loss: MSE between predicted and target params/IC (if available)
    """

    def __init__(self, config: PINNLossConfig) -> None:
        """
        Args:
            config: PINNLossConfig with loss weights
        """
        super().__init__()
        self.config = config
        self.step = 0

    def forward(
        self, input: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total PINN loss.

        Args:
            input: Dictionary with "latent_x", "d_latent_x" and optional "params", "ic" keys
            target: Dictionary with "x", "d_latent_x", "params", "ic" keys
            epoch: Current epoch number (for logging)

        Returns:
            Total loss scalar
        """
        if not isinstance(input, dict) or not isinstance(target, dict):
            raise ValueError("Input and target must be dictionaries")
        loss = {}
        input_copy = input.copy()

        n = target["x"].shape[-2] // 2
        # Use predicted params for inverse PINN, and data for surrogate:
        surrogate_mode = False
        if "params" not in input:
            surrogate_mode = True
            input_copy["params"] = target["params"]
        mu = input_copy["params"][..., n : 2 * n]
        latent_state = input_copy["latent_x"]
        state_components = compute_observables_torch(latent_state, mu, n, smooth=False)
        state = torch.cat(state_components, dim=-2)
        input_copy["x"] = state

        loss["data"] = nn.functional.mse_loss(state, target["x"])

        loss["physics"] = nn.functional.mse_loss(input_copy["d_latent_x"], target["d_latent_x"])

        if not surrogate_mode:
            input_params_ic = torch.cat([input["params"], input["ic"]], dim=1)
            target_params_ic = torch.cat([target["params"], target["ic"]], dim=1)
            loss["supervised"] = nn.functional.mse_loss(input_params_ic, target_params_ic)
        # elif self.config.supervised_weight > 0:
        #     loss["supervised"] = torch.tensor(
        #         0.0, device=state.device, requires_grad=True, dtype=state.dtype
        #     )
        else:
            loss["supervised"] = torch.tensor(
                0.0, device=state.device, requires_grad=False, dtype=state.dtype
            )

        loss["total"] = (
            self.config.data_weight * loss["data"]
            + self.config.physics_weight * loss["physics"]
            + self.config.supervised_weight * loss["supervised"]
        )

        log_PINN_loss(loss, step=self.step)
        self.step += 1

        return loss["total"]
