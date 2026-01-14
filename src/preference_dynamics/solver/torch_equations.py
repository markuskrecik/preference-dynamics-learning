"""
PyTorch differentiable implementations of preference dynamics ODE equations.

This module provides differentiable (PyTorch) versions of the ODE equations
for use in physics-informed neural networks (PINNs).

System equations:
    v'(t) = g - Π · a(t)
    m'(t) = u(t) - Γ · a(t)

where:
    u(t) = max(v(t), μ)  [element-wise max]
    a(t) = max(m(t), 0)  [element-wise max]
"""

import torch


def preference_dynamics_rhs_torch(
    t: torch.Tensor,
    state: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RHS of preference dynamics assuming state shape (..., 2n, T).

    Args:
        t: Time points of shape (..., T). Not used in computation.
        state: Latent state trajectory (..., 2n, T).
        params: Parameters [g, μ, Π, Γ] of shape (..., 2n + 2n²).

    Returns:
        Time derivative of latent state [v'(t), m'(t)] with shape (..., 2n, T).
    """
    n = state.shape[-2] // 2
    if state.shape[-2] != 2 * n:
        raise ValueError(
            f"Expected state dim -2 shape to be 2*n_actions={2 * n}, got {state.shape[-2]}"
        )

    g = params[:n]
    mu = params[n : 2 * n]
    Pi_flat = params[2 * n : 2 * n + n**2]
    Gamma_flat = params[2 * n + n**2 :]
    Pi = Pi_flat.view(n, n)
    Gamma = Gamma_flat.view(n, n)

    g_aligned = g.unsqueeze(-1)

    u, a = compute_observables_torch(state, mu, n)

    v_dot = g_aligned - torch.einsum("...ij,...jt->...it", Pi, a)
    m_dot = u - torch.einsum("...ij,...jt->...it", Gamma, a)

    return torch.cat([v_dot, m_dot], dim=-2)


def _compute_observables_non_smooth_torch(
    state: torch.Tensor,
    mu: torch.Tensor,
    n_actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = state[..., :n_actions, :]
    m = state[..., n_actions:, :]

    mu_aligned = mu.unsqueeze(-1)

    u = torch.maximum(v, mu_aligned)
    a = torch.maximum(m, torch.zeros_like(m))
    return (u, a)


def _compute_observables_smooth_torch(
    state: torch.Tensor,
    mu: torch.Tensor,
    n_actions: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = state[..., :n_actions, :]
    m = state[..., n_actions:, :]

    mu_aligned = mu.unsqueeze(-1)

    u = torch.nn.functional.softplus((v - mu_aligned) / temperature) * temperature + mu_aligned
    a = torch.nn.functional.softplus(m / temperature) * temperature

    return (u, a)


def compute_observables_torch(
    state: torch.Tensor,
    mu: torch.Tensor,
    n_actions: int,
    smooth: bool = False,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute observed variables (u, a) with optional smooth approximation.
    Supports batched time arrays: state must have shape (..., 2n, T), mu should be (..., n).

    Args:
        state: Latent state [v_1, ..., v_n, m_1, ..., m_n] shape (..., 2n, T)
        mu: Desire thresholds [μ_1, ..., μ_n] shape (..., n)
        n_actions: Number of actions (n)
        smooth: Whether to use smooth approximation (default: False)
        temperature: Temperature parameter for smooth approximation (default: 1.0)

    Returns:
        Tuple (u, a) with shape (..., n, T) each.
    """
    if smooth:
        return _compute_observables_smooth_torch(state, mu, n_actions, temperature)
    else:
        return _compute_observables_non_smooth_torch(state, mu, n_actions)
