"""
Stability analysis for preference dynamics ODE system.

This module provides functions to check stability criteria for ODE configurations
based on eigenvalue analysis of system matrices.
"""

from typing import Literal

import numpy as np

from preference_dynamics.schemas import ParameterVector, StabilityReport
from preference_dynamics.solver.equations import compute_equilibria

StabilityMethod = Literal["eigenvalues", "none"]


def _check_stability_eigenvalues(parameters: ParameterVector) -> tuple[bool, list[StabilityReport]]:
    """
    Check stability of equilibria by computing eigenvalues and eigenvectors of Jacobian.

    Computes equilibria, then for each equilibrium computes the Jacobian and its
    eigenvalues. Considers an equilibrium stable if no positive eigenvalues have
    eigenvectors indicating growth to +infinity, or if the magnitude of the most
    negative eigenvalue exceeds the most positive eigenvalue.

    Args:
        parameters: ParameterVector to check

    Returns:
        Tuple of:
            - bool: True if all equilibria are stable
            - list[StabilityReport]: Reports for each equilibrium

    Note:
        Edge case EV=0 is considered stable. May have some rare false negatives.

    TODO:
        - Currently ignores DP. Check if DP is relevant for Jacobian & stability.
    """
    equilibria = compute_equilibria(parameters)
    n_actions = parameters.n_actions

    results = []
    for e in equilibria:
        P = e["actions"]
        n_P = len(P)
        Pi_P = parameters.Pi[np.ix_(P, P)]
        Gamma_P = parameters.Gamma[np.ix_(P, P)]
        identity_P = np.eye(n_P)
        zero_P = np.zeros((n_P, n_P))
        jacobian = np.block([[zero_P, -Pi_P], [identity_P, -Gamma_P]])
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)

        # Usually EV > 0 <=> unstable.
        # Here: Only if actions also grow to +infinity (eigenvector > 0), since bounded from below.
        # If actions "grow to -infinity", then stable limit cycle.
        unstable_eigenvalues = [
            # TODO: any or all? any more reasonable, but reports more false negatives
            np.all(np.real(evec[n_actions:]) > 0)
            for ev, evec in zip(eigenvalues, eigenvectors.T, strict=True)
            if np.real(ev) > 0
        ]
        # TODO: Check if this condition sufficient:
        stable_eigenvalues_larger_than_unstable = np.abs(eigenvalues.min()) > eigenvalues.max()
        is_stable = not np.any(unstable_eigenvalues) or stable_eigenvalues_larger_than_unstable

        results.append(
            StabilityReport(
                is_stable=is_stable,
                actions=P,
                desired_actions=e["desired_actions"],
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
            )
        )
    # TODO: Decide on all vs any:
    # If all: Also filter out bistable cases with stable and unstable equilibria.
    all_stable = all(r.is_stable for r in results) if results else False

    return all_stable, results


def check_stability(parameters: ParameterVector, method: StabilityMethod = "eigenvalues") -> bool:
    """
    Check stability criteria for preference dynamics ODE configuration.

    Args:
        parameters: ParameterVector to check
        method: Method to use for stability checking ("eigenvalues" or "none").
            "none" always returns True.

    Returns:
        True if equilibria are stable (or method="none"), False otherwise

    Raises:
        ValueError: If invalid method is provided
    """
    if method == "eigenvalues":
        is_stable, results = _check_stability_eigenvalues(parameters)
    elif method == "none":
        is_stable = True
    else:
        raise ValueError(f"Invalid method: {method}")
    return is_stable
