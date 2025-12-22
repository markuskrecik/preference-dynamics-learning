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
    n_actions: int,
    g: torch.Tensor,
    mu: torch.Tensor,
    Pi: torch.Tensor,
    Gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        t: currently only used for shape inference:
            if len(t) > 1, then state has time dimension
        state: (B, T, 2n)
        n_actions: int
        g: (B, n)
        mu: (B, n)
        Pi: (B, n, n)
        Gamma: (B, n, n)

    Note:
    - B & T are optional

    Assumes:
    - g, mu, Pi, Gamma don't have time dimension
    - batch dimensions of all match
    - state with either batch or time dimension (but not both):
        Ambiguous case resolved through shape of t
    """
    state_dims = state.ndim

    if state_dims == 3:
        batch_present = True
        time_present = True
    elif state_dims == 2:
        # solve ambiguous case through time tensor
        if t.shape[0] > 1:
            batch_present = False
            time_present = True
        else:
            batch_present = True
            time_present = False
    elif state_dims == 1:
        batch_present = False
        time_present = False
    else:
        raise ValueError(f"Invalid dimensions of state: {state_dims}")

    state = state.clone()
    g = g.clone()
    mu = mu.clone()
    Pi = Pi.clone()
    Gamma = Gamma.clone()

    if not batch_present:
        state.unsqueeze_(0)
        g.unsqueeze_(0)
        mu.unsqueeze_(0)
        Pi.unsqueeze_(0)
        Gamma.unsqueeze_(0)

    if not time_present:
        state.unsqueeze_(1)
    # Params assumed to not have time dimension:
    g.unsqueeze_(1)
    mu.unsqueeze_(1)
    Pi.unsqueeze_(1)
    Gamma.unsqueeze_(1)

    shapes = [var.shape for var in (state, g, mu, Pi, Gamma)]
    if not all(shape[0] == shapes[0][0] for shape in shapes):
        raise ValueError(f"Batch dimensions of inputs don't match: {shapes}")
    # TODO: check for n_actions consistency: (..., n) or (..., n, n)
    # TODO: check for ndims consistency: 3 or 4 (Pi, Gamma)

    u, a = compute_observables_torch(state, mu, n_actions)

    v_dot = g - torch.einsum("...ij,...j->...i", Pi, a)
    m_dot = u - torch.einsum("...ij,...j->...i", Gamma, a)

    result = torch.cat([v_dot, m_dot], dim=-1)

    return result


def _compute_observables_non_smooth_torch(
    state: torch.Tensor,
    mu: torch.Tensor,
    n_actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = state[..., :n_actions]
    m = state[..., n_actions:]

    u = torch.maximum(v, mu)
    zero = torch.zeros_like(m)
    a = torch.maximum(m, zero)
    return (u, a)


def _compute_observables_smooth_torch(
    state: torch.Tensor,
    mu: torch.Tensor,
    n_actions: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = state[..., :n_actions]
    m = state[..., n_actions:]

    u = torch.nn.functional.softplus((v - mu) / temperature) * temperature + mu
    a = torch.nn.functional.softplus(m / temperature) * temperature

    return (u, a)


def compute_observables_torch(
    state: torch.Tensor,
    mu: torch.Tensor,
    n_actions: int,
    smooth: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute observed variables (u, a) with optional smooth approximation.

    Supports batched time arrays: if state has shape (..., T, 2n), mu should be (..., T, n) or (..., 1, n).

    Args:
        state: Latent state [v_1, ..., v_n, m_1, ..., m_n] shape (..., 2n) or (..., T, 2n)
        mu: Desire thresholds [μ_1, ..., μ_n] shape (..., n) or (..., T, n) or (..., 1, n)
        n_actions: Number of actions (n)
        smooth: Whether to use smooth approximation (default: False)
        temperature: Temperature parameter for smooth approximation (default: 1.0)

    Returns:
        observables: [u_1, ..., u_n, a_1, ..., a_n] (..., T, 2n)
    """
    if smooth:
        return _compute_observables_smooth_torch(state, mu, n_actions, temperature)
    else:
        return _compute_observables_non_smooth_torch(state, mu, n_actions)
