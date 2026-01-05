"""
Loss computation utilities.
"""

import torch
from torch import nn

from preference_dynamics.schemas import PINNLossConfig
from preference_dynamics.solver.torch_equations import (
    compute_observables_torch,
    preference_dynamics_rhs_torch,
)

# def compute_physics_residual_score(
#     surrogate_model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
#     time_points: torch.Tensor,
#     # TODO: either supply params, ic, or vecs and matrices
#     params: ParameterVector,
#     ic: ICVector,
# ) -> torch.Tensor:
#     """
#     Compute physics residual score using vectorized autodiff.

#     The physics residual is computed as the squared L2 norm of the ODE residual:
#         R(t) = ||d(state)/dt - RHS(t, state(t), params)||²

#     This version expects surrogate_model to accept batched time arrays and return
#     batched trajectories: surrogate_model(time_points, params, ic) -> (..., T, 2n)

#     Args:
#         surrogate_model: Function that takes time tensor of shape (T,) and returns
#             state trajectory of shape (..., T, 2n). Must support autodiff.
#         time_points: Time points of shape (T,) where T is the number of collocation points
#         n_actions: Number of actions (n)
#         g: Drive vector of shape (..., n)
#         mu: Desire thresholds of shape (..., n)
#         Pi: Coupling matrix Π of shape (..., n, n)
#         Gamma: Damping matrix Γ of shape (..., n, n)

#     Returns:
#         Physics residual score of shape (...,) (scalar per batch element)
#     """
#     if not time_points.requires_grad:
#         time_points = time_points.requires_grad_(True)

#     params_t = torch.tensor(
#         params,
#         dtype=time_points.dtype,
#         device=time_points.device,
#         requires_grad=True,
#         pin_memory=True,
#     )
#     ic_t = torch.tensor(
#         ic, dtype=time_points.dtype, device=time_points.device, requires_grad=True, pin_memory=True
#     )

#     # Compute jacobian: d(state)/dt for each time point
#     # jacobian returns shape (..., T, 2n, T) where [..., t, :, t] is d(state[t])/dt[t]
#     def state_fn(t: torch.Tensor) -> torch.Tensor:
#         return surrogate_model(t, params, ic)

#     T = time_points.shape[0]

#     try:
#         # autograd over all times and channels:
#         state_jacobian = jacobian(
#             state_fn,
#             time_points,
#             create_graph=True,
#             vectorize=True,
#         )
#         # Extract diagonal: d(state[t])/dt[t] for each t
#         # jacobian shape is (..., T, 2n, T), we want (..., T, 2n)
#         state_derivative_actual = torch.stack(
#             [state_jacobian[..., t, :, t] for t in range(T)], dim=-2
#         )  # (..., T, 2n)
#     except Exception as e:
#         raise e
#         # state_derivative_actual_list = []
#         # for t_idx in range(T):
#         #     t_tensor = time_points[t_idx].detach().clone().requires_grad_(True)
#         #     state_t = surrogate_model(t_tensor)  # (..., 2n)

#         #     grad_t = torch.zeros_like(state_t)
#         #     for i in range(state_t.shape[-1]):
#         #         grad_outputs = torch.zeros_like(state_t)
#         #         grad_outputs[..., i] = 1.0
#         #         grad_i = torch.autograd.grad(
#         #             outputs=state_t,
#         #             inputs=t_tensor,
#         #             grad_outputs=grad_outputs,
#         #             create_graph=True,
#         #             retain_graph=True,
#         #         )[0]
#         #         grad_t[..., i] = grad_i

#         #     state_derivative_actual_list.append(grad_t)

#         # state_derivative_actual = torch.stack(state_derivative_actual_list, dim=-2)  # (..., T, 2n)

#     state_trajectory = surrogate_model(time_points, params_t, ic_t)  # (..., T, 2n)

#     params_dict = {
#         k: torch.tensor(
#             v,
#             dtype=time_points.dtype,
#             device=time_points.device,
#             requires_grad=True,
#             pin_memory=True,
#         )
#         for k, v in params.to_dict().items()
#     }

#     state_derivative_pred = preference_dynamics_rhs_torch(
#         t=time_points,
#         state=state_trajectory,
#         **params_dict,
#     )  # (..., T, 2n)

#     residual = state_derivative_pred - state_derivative_actual
#     residual_norm_sq = torch.sum(residual**2, dim=-1)  # (..., T)
#     return torch.mean(residual_norm_sq, dim=-1)  # (...,)


# def compute_inverse_pinn_loss(
#     model: torch.nn.Module,
#     observed_trajectory: torch.Tensor,
#     observed_time_points: torch.Tensor,
#     objective_config: ObjectiveConfig,
#     n_actions: int,
#     collocation_time_points: torch.Tensor | None = None,
#     target_params: torch.Tensor | None = None,
#     target_ic: torch.Tensor | None = None,
# ) -> dict[str, torch.Tensor]:
#     """
#     Compute inverse PINN loss with multi-term objective.

#     The loss consists of:
#         - Data loss: discrepancy between observed and predicted observables
#         - Physics loss: ODE residual on predicted state
#         - Supervised loss: parameter/IC prediction error (if labels available)

#     Args:
#         model: Inverse PINN model (InversePINNPredictor)
#         observed_trajectory: Observed time series [u, a] of shape (B, 2n, T)
#         observed_time_points: Observed time points of shape (T,)
#         objective_config: Objective configuration with weights and collocation strategy
#         n_actions: Number of actions (n)
#         collocation_time_points: Collocation points for physics residual (if None, uses strategy)
#         target_params: Ground truth parameters of shape (B, 2n + 2n²) (optional)
#         target_ic: Ground truth initial conditions of shape (B, 2n) (optional)

#     Returns:
#         Dictionary with:
#             - total_loss: Weighted sum of all loss terms
#             - data_loss: Data fitting loss
#             - physics_loss: Physics residual loss
#             - supervised_loss: Supervised parameter/IC loss (if labels available)
#     """
#     outputs = model(observed_trajectory, observed_time_points, return_intermediates=True)
#     predicted_observables = outputs["observables"]
#     params_constrained = outputs["params_constrained"]
#     ic = outputs["ic"]

#     data_loss = torch.nn.functional.mse_loss(
#         predicted_observables, observed_trajectory, reduction="mean"
#     )

#     if collocation_time_points is None:
#         time_span = (float(observed_time_points[0]), float(observed_time_points[-1]))
#         if objective_config.collocation == "observed_times":
#             collocation_time_points = observed_time_points
#         elif objective_config.collocation == "observed_plus_extra":
#             n_extra = len(observed_time_points)
#             collocation_time_points = get_collocation_points(
#                 observed_time_points,
#                 time_span,
#                 strategy="uniform",
#                 n_collocation_points=n_extra,
#             )
#             collocation_time_points = torch.from_numpy(collocation_time_points).to(
#                 observed_trajectory.device
#             )
#         else:
#             raise ValueError(f"Unknown collocation strategy: {objective_config.collocation}")

#     physics_loss = _compute_physics_residual_batched(
#         model=model,
#         collocation_time_points=collocation_time_points,
#         params_constrained=params_constrained,
#         ic=ic,
#         n_actions=n_actions,
#         smooth=not objective_config.non_smooth_observables,
#         temperature=objective_config.smoothness_temperature or 1.0,
#     )

#     supervised_loss = None
#     if target_params is not None and target_ic is not None:
#         params_loss = torch.nn.functional.mse_loss(
#             params_constrained, target_params, reduction="mean"
#         )
#         ic_loss = torch.nn.functional.mse_loss(ic, target_ic, reduction="mean")
#         supervised_loss = params_loss + ic_loss
#     elif objective_config.supervised_weight > 0:
#         supervised_loss = torch.tensor(0.0, device=observed_trajectory.device, requires_grad=True)

#     total_loss = (
#         objective_config.data_weight * data_loss + objective_config.physics_weight * physics_loss
#     )
#     if supervised_loss is not None:
#         total_loss = total_loss + objective_config.supervised_weight * supervised_loss

#     result = {
#         "total_loss": total_loss,
#         "data_loss": data_loss,
#         "physics_loss": physics_loss,
#     }
#     if supervised_loss is not None:
#         result["supervised_loss"] = supervised_loss

#     return result


# def _compute_physics_residual_batched(
#     model: torch.nn.Module,
#     collocation_time_points: torch.Tensor,
#     params_constrained: torch.Tensor,
#     ic: torch.Tensor,
#     n_actions: int,
#     smooth: bool = False,
#     temperature: float = 1.0,
# ) -> torch.Tensor:
#     """
#     Compute physics residual for batched inputs.

#     Args:
#         model: Inverse PINN model
#         collocation_time_points: Collocation points of shape (T_colloc,)
#         params_constrained: Constrained parameters of shape (B, 2n + 2n²)
#         ic: Initial conditions of shape (B, 2n)
#         n_actions: Number of actions (n)
#         smooth: Whether to use smooth observables
#         temperature: Temperature for smooth approximation

#     Returns:
#         Physics residual loss of shape (B,)
#     """
#     B = params_constrained.shape[0]
#     T_colloc = collocation_time_points.shape[0]

#     if not collocation_time_points.requires_grad:
#         collocation_time_points = collocation_time_points.requires_grad_(True)

#     t_batch = collocation_time_points.unsqueeze(0).expand(B, -1)

#     state = model.surrogate(t_batch, params_constrained, ic)
#     state_T = state.transpose(1, 2)

#     g = params_constrained[:, :n_actions]
#     mu = params_constrained[:, n_actions : 2 * n_actions]
#     Pi_flat = params_constrained[:, 2 * n_actions : 2 * n_actions + n_actions**2]
#     Gamma_flat = params_constrained[:, 2 * n_actions + n_actions**2 :]
#     Pi = Pi_flat.reshape(B, n_actions, n_actions)
#     Gamma = Gamma_flat.reshape(B, n_actions, n_actions)

#     mu_expanded = mu.unsqueeze(1).expand(B, T_colloc, -1)

#     state_derivative_actual = _compute_state_derivative(
#         state_T, collocation_time_points, B, T_colloc, n_actions
#     )

#     observables = compute_observables_torch(
#         state=state_T,
#         mu=mu_expanded,
#         n_actions=n_actions,
#         smooth=smooth,
#         temperature=temperature,
#     )

#     u = observables[..., :n_actions]
#     a = observables[..., n_actions:]

#     g_expanded = g.unsqueeze(1).expand(B, T_colloc, -1)
#     v_dot = g_expanded - torch.einsum("bij,bj->bi", Pi, a)
#     m_dot = u - torch.einsum("bij,bj->bi", Gamma, a)

#     state_derivative_pred = torch.cat([v_dot, m_dot], dim=-1)

#     residual = state_derivative_pred - state_derivative_actual
#     residual_norm_sq = torch.sum(residual**2, dim=-1)
#     return torch.mean(residual_norm_sq, dim=-1)


# def _compute_state_derivative(
#     state: torch.Tensor,
#     time_points: torch.Tensor,
#     B: int,
#     T: int,
#     n_actions: int,
# ) -> torch.Tensor:
#     """
#     Compute d(state)/dt using autograd.

#     Args:
#         state: State trajectory of shape (B, T, 2n)
#         time_points: Time points of shape (T,) with requires_grad=True
#         B: Batch size
#         T: Number of time points
#         n_actions: Number of actions (n)

#     Returns:
#         State derivative of shape (B, T, 2n)
#     """
#     state_derivative_list = []
#     for t_idx in range(T):
#         state_t = state[:, t_idx, :]

#         grad_outputs = torch.ones_like(state_t)
#         grad_t = torch.autograd.grad(
#             outputs=state_t,
#             inputs=time_points,
#             grad_outputs=grad_outputs,
#             create_graph=True,
#             retain_graph=True,
#             allow_unused=True,
#         )[0]

#         if grad_t is None:
#             grad_t = torch.zeros(B, 2 * n_actions, device=state.device)
#         elif grad_t.ndim == 1:
#             grad_t = grad_t.unsqueeze(0).expand(B, -1)
#         elif grad_t.shape[0] == 1:
#             grad_t = grad_t.expand(B, -1)

#         state_derivative_list.append(grad_t)

#     return torch.stack(state_derivative_list, dim=1)


class PINNLoss(nn.Module):  # type: ignore
    """
    Physics-Informed Neural Network loss for inverse PINN training.

    Computes:
        - Data loss: MSE between predicted and target state
        - Physics loss: ODE residual (d(state)/dt - RHS)
        - Supervised loss: MSE between predicted and target params/IC (if available)
    """

    def __init__(self, config: PINNLossConfig) -> None:
        """
        Args:
            config: PINNLossConfig with loss weights
        """
        super().__init__()
        self.config = config

    def physics_loss(
        self, input: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physics residual loss.

        Args:
            input: Dictionary with "x" key (B, 2n, T)
            target: Dictionary with "t" key (B, T) with requires_grad=True

        Returns:
            Physics loss scalar
        """
        time_points = input["t"]
        state = input["x"]

        if time_points.shape != state[..., 0, :].shape:
            raise ValueError(
                f"Time points {time_points.shape} and state {state.shape} must share batch and time dimensions."
            )
        B, channels, T = state.shape
        n = channels // 2

        state_T = state.transpose(1, 2)  # (B, T, 2n)

        # d_latent_state_T = torch.zeros_like(state_T)
        d_latent_state = []
        for j in range(state_T.shape[-1]):
            grad_j = torch.autograd.grad(
                outputs=state_T[..., j],  # (B, T)
                inputs=time_points,  # (B, T)
                grad_outputs=torch.ones_like(state_T[..., j]),
                retain_graph=True,
                create_graph=True,
            )[0]
            d_latent_state.append(grad_j)
            # d_latent_state_T[..., j] = grad_j

            # d_latent_state = []
            # for t_idx in range(T):
            #     state_t = state_T[:, t_idx, :]
            #     grad_outputs = torch.ones_like(state_t)
            #     grad_t = torch.autograd.grad(
            #         outputs=state_t,
            #         inputs=time_points,
            #         grad_outputs=grad_outputs,
            #         # is_grads_batched=True,
            #         create_graph=True,
            #         retain_graph=True,
            #         allow_unused=True,
            #     )[0]

            # if grad_t is None:
            # grad_t = torch.zeros(B, channels, device=state.device)
            # elif grad_t.ndim == 1:
            #     grad_t = grad_t.unsqueeze(0).expand(B, -1)
            # elif grad_t.shape[0] == 1:
            #     grad_t = grad_t.expand(B, -1)

            # d_latent_state.append(grad_t)
        d_latent_state_T = torch.stack(d_latent_state, dim=-1)
        # d_latent_state = d_latent_state_T.transpose(1, 2)

        params = input["params"]
        g = params[..., :n]
        mu = params[..., n : 2 * n]
        Pi_flat = params[..., 2 * n : 2 * n + n**2]
        Gamma_flat = params[..., 2 * n + n**2 :]
        Pi = Pi_flat.reshape(B, n, n)
        Gamma = Gamma_flat.reshape(B, n, n)

        mu_expanded = mu.unsqueeze(1).expand(B, T, -1)
        g_expanded = g.unsqueeze(1).expand(B, T, -1)

        d_latent_state_T_pred = preference_dynamics_rhs_torch(
            t=time_points,
            state=state_T,
            n_actions=n,
            g=g_expanded,
            mu=mu_expanded,
            Pi=Pi,
            Gamma=Gamma,
        )

        return nn.functional.mse_loss(d_latent_state_T, d_latent_state_T_pred)

    def data_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss.

        Args:
            input: Predicted state of shape (B, 2n, T)
            target: Target state of shape (B, 2n, T)

        Returns:
            Data loss scalar
        """
        return nn.functional.mse_loss(input, target)

    def supervised_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised parameter/IC loss.

        Args:
            input: Predicted params+IC concatenated of shape (B, n_params + n_ic)
            target: Target params+IC concatenated of shape (B, n_params + n_ic)

        Returns:
            Supervised loss scalar
        """
        return nn.functional.mse_loss(input, target)

    def forward(
        self, input: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total PINN loss.

        Args:
            input: Dictionary with "latent_x" and optional "params", "ic" keys
            target: Dictionary with "x", "t", "params", "ic" keys
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
        B, channels, T = latent_state.shape
        latent_state_T = latent_state.transpose(1, 2)
        mu_expanded = mu.unsqueeze(-1).expand(B, T, -1)

        state_T = compute_observables_torch(latent_state_T, mu_expanded, n, smooth=False)
        state = torch.cat(state_T, dim=-1).transpose(1, 2)
        input_copy["x"] = state

        loss["data"] = self.data_loss(state, target["x"])

        loss["physics"] = self.physics_loss(input_copy, target)

        if not surrogate_mode:
            input_params_ic = torch.cat([input["params"], input["ic"]], dim=1)
            target_params_ic = torch.cat([target["params"], target["ic"]], dim=1)
            loss["supervised"] = self.supervised_loss(input_params_ic, target_params_ic)
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

        # log_PINN_loss(loss, step=epoch)

        return loss["total"]
