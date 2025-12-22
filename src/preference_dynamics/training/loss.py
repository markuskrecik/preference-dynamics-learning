"""
Loss computation utilities.
"""

from collections.abc import Callable

import torch
from torch.autograd.functional import jacobian

from preference_dynamics.schemas import ICVector, ParameterVector
from preference_dynamics.solver.torch_equations import preference_dynamics_rhs_torch


def compute_physics_residual_score(
    surrogate_model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    time_points: torch.Tensor,
    # TODO: either supply params, ic, or vecs and matrices
    params: ParameterVector,
    ic: ICVector,
) -> torch.Tensor:
    """
    Compute physics residual score using vectorized autodiff.

    The physics residual is computed as the squared L2 norm of the ODE residual:
        R(t) = ||d(state)/dt - RHS(t, state(t), params)||²

    This version expects surrogate_model to accept batched time arrays and return
    batched trajectories: surrogate_model(time_points, params, ic) -> (..., T, 2n)

    Args:
        surrogate_model: Function that takes time tensor of shape (T,) and returns
            state trajectory of shape (..., T, 2n). Must support autodiff.
        time_points: Time points of shape (T,) where T is the number of collocation points
        n_actions: Number of actions (n)
        g: Drive vector of shape (..., n)
        mu: Desire thresholds of shape (..., n)
        Pi: Coupling matrix Π of shape (..., n, n)
        Gamma: Damping matrix Γ of shape (..., n, n)

    Returns:
        Physics residual score of shape (...,) (scalar per batch element)
    """
    if not time_points.requires_grad:
        time_points = time_points.requires_grad_(True)

    params_t = torch.tensor(
        params,
        dtype=time_points.dtype,
        device=time_points.device,
        requires_grad=True,
        pin_memory=True,
    )
    ic_t = torch.tensor(
        ic, dtype=time_points.dtype, device=time_points.device, requires_grad=True, pin_memory=True
    )

    # Compute jacobian: d(state)/dt for each time point
    # jacobian returns shape (..., T, 2n, T) where [..., t, :, t] is d(state[t])/dt[t]
    def state_fn(t: torch.Tensor) -> torch.Tensor:
        return surrogate_model(t, params, ic)

    T = time_points.shape[0]

    try:
        # autograd over all times and channels:
        state_jacobian = jacobian(
            state_fn,
            time_points,
            create_graph=True,
            vectorize=True,
        )
        # Extract diagonal: d(state[t])/dt[t] for each t
        # jacobian shape is (..., T, 2n, T), we want (..., T, 2n)
        state_derivative_actual = torch.stack(
            [state_jacobian[..., t, :, t] for t in range(T)], dim=-2
        )  # (..., T, 2n)
    except Exception as e:
        raise e
        # state_derivative_actual_list = []
        # for t_idx in range(T):
        #     t_tensor = time_points[t_idx].detach().clone().requires_grad_(True)
        #     state_t = surrogate_model(t_tensor)  # (..., 2n)

        #     grad_t = torch.zeros_like(state_t)
        #     for i in range(state_t.shape[-1]):
        #         grad_outputs = torch.zeros_like(state_t)
        #         grad_outputs[..., i] = 1.0
        #         grad_i = torch.autograd.grad(
        #             outputs=state_t,
        #             inputs=t_tensor,
        #             grad_outputs=grad_outputs,
        #             create_graph=True,
        #             retain_graph=True,
        #         )[0]
        #         grad_t[..., i] = grad_i

        #     state_derivative_actual_list.append(grad_t)

        # state_derivative_actual = torch.stack(state_derivative_actual_list, dim=-2)  # (..., T, 2n)

    state_trajectory = surrogate_model(time_points, params_t, ic_t)  # (..., T, 2n)

    params_dict = {
        k: torch.tensor(
            v,
            dtype=time_points.dtype,
            device=time_points.device,
            requires_grad=True,
            pin_memory=True,
        )
        for k, v in params.to_dict().items()
    }

    state_derivative_pred = preference_dynamics_rhs_torch(
        t=time_points,
        state=state_trajectory,
        **params_dict,
    )  # (..., T, 2n)

    residual = state_derivative_pred - state_derivative_actual
    residual_norm_sq = torch.sum(residual**2, dim=-1)  # (..., T)
    return torch.mean(residual_norm_sq, dim=-1)  # (...,)
