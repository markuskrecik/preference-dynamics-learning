"""
ODE integration for preference dynamics system.
"""

import functools

import numpy as np
from scipy.integrate import solve_ivp

from preference_dynamics.schemas import (
    ODESolverConfig,
    TimeSeriesSample,
)
from preference_dynamics.solver.equations import (
    compute_observables,
    preference_dynamics_rhs,
)


class SolverConvergenceError(Exception):
    """Raised when ODE solver fails to converge."""

    pass


def solve_ODE(
    config: ODESolverConfig,
) -> TimeSeriesSample:
    """
    Solve preference dynamics ODE and generate time series sample.

    This function integrates the preference dynamics ODE system from the
    initial conditions using the specified parameters, then transforms the
    latent state (v, m) to observed variables (u, a).

    Args:
        config: Complete ODESolverConfig with ODE system and solver settings

    Returns:
        TimeSeriesSample with:
            - time_series: shape (2n, T) containing [u, a] (observed variables)
            - time_points: shape (T,) time evaluation points
            - config: the input ODESolverConfig
            - metadata: solver info (nfev, njev, nlu, success, message)

    Raises:
        SolverConvergenceError: If ODE integration fails to converge

    Example:
        >>> from preference_dynamics.schemas import (
        ...     ODEConfig, SolverConfig, ODESolverConfig,
        ...     ParameterVector, ICVector
        ... )
        >>> import numpy as np
        >>> params = ParameterVector(values=np.ones(24))
        >>> ic = ICVector(values=np.random.randn(6))
        >>> ode_config = ODEConfig(
        ...     parameters=params,
        ...     initial_conditions=ic
        ... )
        >>> solver_config = SolverConfig(
        ...     time_span=(0.0, 100.0),
        ...     n_time_points=1000
        ... )
        >>> config = ODESolverConfig(ode=ode_config, solver=solver_config)
        >>> sample = solve_ODE(config)

    Notes:
        - Uses functools.partial to pre-bind parameters for efficiency
    """
    n: int = config.ode.n_actions  # type: ignore

    t_span = config.solver.time_span
    if t_span[1] <= t_span[0]:
        raise ValueError("time_span end must be greater than start")
    t_eval = np.linspace(t_span[0], t_span[1], config.solver.n_time_points)

    # Use functools.partial to pre-bind parameters for efficiency
    # ode_fun = preference_dynamics_rhs_wrapper(config.ode.parameters)
    ode_fun = functools.partial(
        preference_dynamics_rhs,
        **config.ode.parameters.to_dict(),
    )

    solution = solve_ivp(
        fun=ode_fun,
        t_span=t_span,
        y0=config.ode.initial_conditions.values,
        method=config.solver.solver_method,
        t_eval=t_eval,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )

    if not solution.success:
        raise SolverConvergenceError(f"ODE solver failed to converge: {solution.message}")

    if np.any(solution.y[:, -1] > 1e5):
        raise SolverConvergenceError(
            "ODE solution diverged due to unstable parameters: state exceeded threshold (1e5)"
        )

    state_trajectory = solution.y  # Shape: (2n, T)

    mu = config.ode.parameters.mu
    observed_trajectory = np.zeros_like(state_trajectory)

    T = state_trajectory.shape[1]
    for i in range(T):
        observed_trajectory[:, i] = compute_observables(
            state=state_trajectory[:, i],
            mu=mu,
            n_actions=n,
        )

    metadata = {
        "n_function_evaluations": solution.nfev,
        "n_jacobian_evaluations": solution.njev,
        "n_lu_decompositions": solution.nlu,
        "solver_success": solution.success,
        "solver_message": solution.message,
    }

    sample = TimeSeriesSample(
        time_series=observed_trajectory,
        time_points=solution.t,
        config=config.model_dump(),  # type: ignore
        metadata=metadata,
    )
    return sample
