"""
Batch data generation for preference dynamics.
"""

from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from preference_dynamics.schemas import (
    ODEConfig,
    ODESolverConfig,
    SolverConfig,
    TimeSeriesSample,
)
from preference_dynamics.solver.integrator import (
    SolverConvergenceError,
    solve_ODE,
)
from preference_dynamics.solver.sampler import ODEConfigSampler


def _solve_and_validate_single(
    ode_config: ODEConfig,
    solver_config: SolverConfig,
    debug: bool = False,
) -> TimeSeriesSample | None:
    """
    Helper function to solve and validate a single ODE config.

    Args:
        ode_config: ODE configuration with parameters and initial conditions
        solver_config: Solver configuration
        debug: Whether to print debug information on errors

    Returns:
        TimeSeriesSample if successful, None if solver fails or validation fails
    """
    try:
        config = ODESolverConfig(ode=ode_config, solver=solver_config.model_dump())  # type: ignore
        sample = solve_ODE(config)
        return sample
    except (SolverConvergenceError, ValueError) as e:
        if debug:
            print(f"Solution skipped: {e}")
        return None


def generate_batch(
    n_samples: int,
    sampler: ODEConfigSampler,
    solver_config: SolverConfig,
    n_jobs: int = -1,
    show_progress: bool = True,
    debug: bool = False,
) -> list[TimeSeriesSample]:
    """
    Generate batch of samples in parallel with stability filtering.

    This function generates samples until n_samples valid samples are collected.

    Args:
        n_samples: Number of samples to generate
        sampler: ODEConfigSampler for sampling parameters and initial conditions
        solver_config: Solver settings to use for all samples
        n_jobs: Number of parallel jobs (-1 = all CPUs, default: -1)
        show_progress: Whether to show tqdm progress bar (default: True)
        debug: Whether to print debug information (default: False)

    Returns:
        List of TimeSeriesSample objects (length = n_samples)

    Example:
        >>> from preference_dynamics.solver import create_default_sampler
        >>> from preference_dynamics.schemas import SolverConfig
        >>> sampler = create_default_sampler(n_actions=3, random_seed=42)
        >>> solver_config = SolverConfig(
        ...     time_span=(0.0, 100.0),
        ...     n_time_points=1000
        ... )
        >>> samples = generate_batch(
        ...     n_samples=10,
        ...     sampler=sampler,
        ...     solver_config=solver_config
        ... )
    """
    samples: list[TimeSeriesSample] = []
    sampler_it = sampler.as_iterator(debug=debug)
    attempts = 0

    pbar = tqdm(total=n_samples, desc="Generating samples", disable=not show_progress)

    if n_jobs != 1:
        num_workers = n_jobs if n_jobs > 0 else cpu_count()

        while len(samples) < n_samples:
            remaining = n_samples - len(samples)
            # When few tasks remain, schedule at least num_workers tasks
            # and hope that one cycle suffices for completion
            batch_size = min(4 * num_workers, remaining + num_workers)
            attempts += batch_size

            tasks = [
                delayed(_solve_and_validate_single)(next(sampler_it), solver_config, debug)
                for _ in range(batch_size)
            ]

            results = Parallel(n_jobs=n_jobs, prefer=None)(tasks)

            for result in results:
                if result is not None:
                    samples.append(result)
                    pbar.update(1)
                    if len(samples) >= n_samples:
                        break

        samples = samples[:n_samples]

    else:
        while len(samples) < n_samples:
            attempts += 1
            ode_config = next(sampler_it)
            result = _solve_and_validate_single(ode_config, solver_config, debug)
            if result is not None:
                samples.append(result)
                pbar.update(1)
                if len(samples) >= n_samples:
                    break

    if debug:
        print(f"Generated {len(samples)} samples in {attempts} attempts")
    pbar.close()
    return samples
