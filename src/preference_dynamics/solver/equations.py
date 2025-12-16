"""
ODE equations for preference dynamics system.

System equations:
    v'(t) = g - Π · a(t)
    m'(t) = u(t) - Γ · a(t)

where:
    u(t) = max(v(t), μ)  [element-wise max]
    a(t) = max(m(t), 0)  [element-wise max]

The system supports variable dimensions (n actions) with the following structure:
- State: [v_1, ..., v_n, m_1, ..., m_n] in R^(2n)
- Parameters: [g, μ, Π, Γ] in R^(2n + 2n²)
- Observations: [u_1, ..., u_n, a_1, ..., a_n] in R^(2n)
"""

from typing import Literal

import numpy as np
from numpydantic import NDArray

from preference_dynamics.schemas import ParameterVector
from preference_dynamics.utils import get_subsets


def preference_dynamics_rhs(
    t: float,
    state: NDArray[Literal["2-*"], float],
    n_actions: int,
    g: NDArray[Literal["*"], float],
    mu: NDArray[Literal["*"], float],
    Pi: NDArray[Literal["*, *"], float],
    Gamma: NDArray[Literal["*, *"], float],
) -> NDArray[Literal["2-*"], float]:
    """
    Compute derivatives for preference dynamics system.

    System equations:
        v'(t) = g - Π · a(t)
        m'(t) = u(t) - Γ · a(t)

    where u(t) = max(v(t), μ) and a(t) = max(m(t), 0).

    Args:
        t: Time (not used, required by ODE solver interface)
        state: State vector [v_1, ..., v_n, m_1, ..., m_n] of shape (2n,)
        n_actions: Number of actions (n)
        g: Drive vector of shape (n,)
        mu: Desire thresholds of shape (n,)
        Pi: Coupling matrix Π of shape (n, n)
        Gamma: Damping matrix Γ of shape (n, n)

    Returns:
        State derivative [v_1', ..., v_n', m_1', ..., m_n'] of shape (2n,)
    """
    observables = compute_observables(state, mu, n_actions)
    u, a = observables[0:n_actions], observables[n_actions : 2 * n_actions]

    v_dot = g - Pi @ a
    m_dot = u - Gamma @ a

    return np.concatenate([v_dot, m_dot])


def compute_observables(
    state: NDArray[Literal["2-*"], float],
    mu: NDArray[Literal["*"], float],
    n_actions: int,
) -> NDArray[Literal["2-*"], float]:
    """
    Compute observed variables (u, a) from latent state (v, m).

    This helper function transforms the latent state into observable quantities.

    Args:
        state: Latent state [v_1, ..., v_n, m_1, ..., m_n] shape (2n,)
        mu: Desire thresholds [μ_1, ..., μ_n] shape (n,)
        n_actions: Number of actions (n)

    Returns:
        observables: [u_1, ..., u_n, a_1, ..., a_n] shape (2n,)
    """
    v = state[0:n_actions]  # Desire potentials
    m = state[n_actions : 2 * n_actions]  # Effort potentials

    u = np.maximum(v, mu)  # Actual desires
    a = np.maximum(m, 0.0)  # Actual efforts

    return np.concatenate([u, a])


def _check_equilibrium_conditions(
    parameters: ParameterVector,
    a_eq: NDArray[Literal["*"], float],
    u_eq: NDArray[Literal["*"], float],
    P: tuple[int, ...],
    DP: tuple[int, ...],
    FP: tuple[int, ...],
    S: tuple[int, ...],
    debug: bool = False,
) -> bool:
    """
    Check validity conditions for a given equilibrium.
    Skip edge case DS: Pi_DS_P @ a_P == g_DS.

    Returns:
        True if all conditions are satisfied, False otherwise.
    """
    if not np.all(a_eq > 0):
        if debug:
            print(f"Skipped {P=}, {DP=}: Some actions {a_eq=} ≤ 0.")
        return False

    mu_DP = parameters.mu[np.ix_(DP)]
    if not np.all(u_eq > mu_DP):
        if debug:
            print(f"Skipped {P=}, {DP=}: Some desires {u_eq=} ≤ μ={mu_DP}.")
        return False

    Gamma_S_P = parameters.Gamma[np.ix_(S, P)]
    mu_S = parameters.mu[np.ix_(S)]
    if not np.all(Gamma_S_P @ a_eq >= mu_S):
        if debug:
            print(f"Skipped {P=}, {DP=}: Some desires {Gamma_S_P @ a_eq=} < μ={mu_S}.")
        return False

    F = FP + S
    Pi_F_P = parameters.Pi[np.ix_(F, P)]
    g_F = parameters.g[np.ix_(F)]
    if not np.all(Pi_F_P @ a_eq >= g_F):
        if debug:
            print(f"Skipped {P=}, {DP=}: Some desires {Pi_F_P @ a_eq=} < g={g_F}.")
        return False

    return True


def _compute_equilibria_mu_neg_infty(
    parameters: ParameterVector, debug: bool = False
) -> list[dict[str, NDArray[Literal["*"], float]]]:
    """Compute equilibria under assumption that μ = -∞.
    Only for debugging purposes."""
    n: int = parameters.n_actions  # type: ignore
    Pi = parameters.Pi
    Gamma = parameters.Gamma
    g = parameters.g
    subsets = get_subsets(n)
    next(subsets)  # skip empty set

    equilibria = []
    for P in subsets:
        Pi_P = Pi[np.ix_(P, P)]
        g_P = g[np.ix_(P)]
        try:
            Pi_P_inv = np.linalg.inv(Pi_P)
            a_eq = Pi_P_inv @ g_P
            if np.any(a_eq <= 0):
                if debug:
                    print(f"Skipped {P=}: {a_eq=} has actions ≤ 0.")
                continue
            Gamma_P = Gamma[np.ix_(P, P)]
            u_eq = Gamma_P @ a_eq
            equilibria.append(
                {"actions": P, "equilibrium_actions": a_eq, "equilibrium_desires": u_eq}
            )
        except np.linalg.LinAlgError:
            if debug:
                print(f"Skipped {P=}: Pi_P is not invertible.")
            continue
    return equilibria


def _compute_equilibria_general(
    parameters: ParameterVector, debug: bool = False
) -> list[dict[str, NDArray[Literal["*"], float]]]:
    """
    Compute equilibria for general case.

    For each subset of performed actions P (actions with a_eq > 0), tries all partitions into DP and FP:
        - DP: desired actions, where u_eq > μ
        - FP: fulfilled actions, where u_eq ≤ μ

    Solves [Π_{DP,P},Γ_{FP,P}] @ a_P = [g_DP,μ_FP] for a_P, where matrices and vectors are vertically stacked.

    Args:
        parameters: ParameterVector to check
        debug: If True, print debug information

    Returns:
        List of dicts with keys:
            - "actions": subset P of performed actions (tuple of indices)
            - "desired_actions": subset DP of desired performed actions (tuple of indices)
            - "equilibrium_actions": a_eq for actions in P (others = 0)
            - "equilibrium_desires": u_eq for actions in DP (others = mu)

    TODO:
        - fill actions w/ 0 if not in P, desires w/ mu if not in DP
    """
    n: int = parameters.n_actions  # type: ignore
    Pi = parameters.Pi
    Gamma = parameters.Gamma
    g = parameters.g
    mu = parameters.mu
    subsets = get_subsets(n)
    next(subsets)  # skip empty set

    equilibria = []
    for P in subsets:
        S = tuple(set(range(n)).difference(P))
        for DP in get_subsets(P):
            FP = tuple(set(P).difference(DP))

            Pi_DP_P = Pi[np.ix_(DP, P)]
            Gamma_FP_P = Gamma[np.ix_(FP, P)]
            g_DP = g[np.ix_(DP)]
            mu_FP = mu[np.ix_(FP)]

            B = np.vstack([Pi_DP_P, Gamma_FP_P])
            b = np.concatenate([g_DP, mu_FP])
            try:
                # Don't need to worry about reordering, since only columns potentially swapped, but not rows
                a_eq = np.linalg.inv(B) @ b
                Gamma_DP_P = Gamma[np.ix_(DP, P)]
                u_eq = Gamma_DP_P @ a_eq
                if not _check_equilibrium_conditions(parameters, a_eq, u_eq, P, DP, FP, S, debug):
                    continue
                equilibria.append(
                    {
                        "actions": P,
                        "desired_actions": DP,
                        "equilibrium_actions": a_eq,
                        "equilibrium_desires": u_eq,
                    }
                )
            except np.linalg.LinAlgError:
                if debug:
                    print(f"Skipped {P=}, {DP=}: B is not invertible.")
                continue

    return equilibria


def compute_equilibria(
    parameters: ParameterVector, debug: bool = False
) -> list[dict[str, NDArray[Literal["*"], float]]]:
    """
    Compute equilibria for general case.

    For each subset of performed actions P (actions with a_eq > 0), tries all partitions into DP and FP:
        - DP: desired actions, where u_eq > μ
        - FP: fulfilled actions, where u_eq ≤ μ

    Solves [Π_{DP,P},Γ_{FP,P}] @ a_P = [g_DP,μ_FP], where matrices are vertically stacked.

    Args:
        parameters: ParameterVector to check
        debug: If True, print debug information

    Returns:
        List of dicts with keys:
            - "actions": subset P (tuple of indices)
            - "desired_actions": subset DP (tuple of indices)
            - "equilibrium_actions": a_eq for actions in P (others = 0)
            - "equilibrium_desires": u_eq for actions in DP (others = mu)
    """

    return _compute_equilibria_general(parameters, debug)
