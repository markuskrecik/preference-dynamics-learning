"""
Parameter sampling utilities for ODE configuration generation.
"""

from collections.abc import Iterator
from typing import Literal

import numpy as np

from preference_dynamics.schemas import ICVector, ODEConfig, ParameterVector
from preference_dynamics.solver.stability import StabilityMethod, check_stability


class ODEConfigSampler:
    """
    Sampler for generating ODEConfig objects with random parameters and initial conditions.

    Args:
        parameter_ranges: Tuple of two ParameterVector instances defining min and max for parameters
        ic_range: Tuple of two ICVector instances defining min and max for initial conditions
        random_seed: Random seed for reproducibility, default: 42
        distribution: Distribution type ('uniform' or 'normal'), default: "uniform"
        stability_method: Method to use for stability checking (StabilityMethod), default: "eigenvalues"

    Raises:
        ValueError: If parameter_ranges and ic_range n_actions don't match
    """

    def __init__(
        self,
        parameter_ranges: tuple[ParameterVector, ParameterVector],
        ic_range: tuple[ICVector, ICVector],
        random_seed: int = 42,
        distribution: Literal["uniform", "normal"] = "uniform",
        stability_method: StabilityMethod = "eigenvalues",
    ):
        self.parameter_ranges = parameter_ranges
        self.ic_range = ic_range
        self.random_seed = random_seed
        self.distribution = distribution
        self.stability_method = stability_method

        param_min, param_max = parameter_ranges
        ic_min, ic_max = ic_range

        if param_min.n_actions != param_max.n_actions:
            raise ValueError(
                f"Parameter range n_actions mismatch: min={param_min.n_actions}, "
                f"max={param_max.n_actions}"
            )

        if ic_min.n_actions != ic_max.n_actions:
            raise ValueError(
                f"IC range n_actions mismatch: min={ic_min.n_actions}, max={ic_max.n_actions}"
            )

        if param_min.n_actions != ic_min.n_actions:
            raise ValueError(
                f"Parameter and IC n_actions mismatch: params={param_min.n_actions}, "
                f"ic={ic_min.n_actions}"
            )

        self.n_actions = param_min.n_actions
        self.rng = np.random.RandomState(random_seed)

    def _sample(self) -> ODEConfig:
        """
        Sample a single ODEConfig with random parameters and initial conditions.

        Returns:
            ODEConfig with randomly sampled parameters and ICs

        Raises:
            ValueError: For unknown distribution
        """
        param_min, param_max = self.parameter_ranges
        ic_min, ic_max = self.ic_range

        # Sample parameters
        if self.distribution == "uniform":
            param_values = self.rng.uniform(
                low=param_min.values, high=param_max.values, size=param_min.values.shape
            )
        elif self.distribution == "normal":
            # For normal distribution, use ranges as mean ± std
            mean = (param_min.values + param_max.values) / 2
            std = (param_max.values - param_min.values) / 4  # ~95% within range
            param_values = self.rng.normal(loc=mean, scale=std, size=param_min.values.shape)
            # Clip to ensure within range
            param_values = np.clip(param_values, param_min.values, param_max.values)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Sample initial conditions
        if self.distribution == "uniform":
            ic_values = self.rng.uniform(
                low=ic_min.values, high=ic_max.values, size=ic_min.values.shape
            )
        elif self.distribution == "normal":
            mean = (ic_min.values + ic_max.values) / 2
            std = (ic_max.values - ic_min.values) / 4
            ic_values = self.rng.normal(loc=mean, scale=std, size=ic_min.values.shape)
            ic_values = np.clip(ic_values, ic_min.values, ic_max.values)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        params = ParameterVector(values=param_values)
        ic = ICVector(values=ic_values)

        return ODEConfig(
            parameters=params.model_dump(),  # type: ignore
            initial_conditions=ic,
        )

    def sample(self, method: StabilityMethod | None = None, debug: bool = False) -> ODEConfig:
        """
        Sample a single stable ODEConfig.

        Repeatedly samples configurations until a stable one is found (based on
        stability checking method). May take multiple attempts.

        Args:
            method: Method to use for stability checking (StabilityMethod). Uses
                self.stability_method if None.
            debug: Whether to print debug information

        Returns:
            ODEConfig with stable parameters and initial conditions
        """
        method = method or self.stability_method

        attempts = 0
        while True:
            attempts += 1
            config = self._sample()
            is_stable = check_stability(config.parameters, method)
            if is_stable:
                if debug:
                    print(f"Stable config found after {attempts} attempts")
                return config

    def as_iterator(
        self, method: StabilityMethod | None = None, debug: bool = False
    ) -> Iterator[ODEConfig]:
        """
        Return an infinite iterator over stable ODEConfig objects.

        Args:
            method: Method to use for stability checking (StabilityMethod). Uses
                self.stability_method if None.
            debug: Whether to print debug information

        Yields:
            Stable ODEConfig objects
        """
        method = method or self.stability_method

        while True:
            yield self.sample(method, debug)

    def sample_batch(self, n_samples: int) -> list[ODEConfig]:
        """
        Sample multiple stable ODEConfig objects.

        Args:
            n_samples: Number of stable configs to sample

        Returns:
            List of stable ODEConfig objects
        """
        configs = []
        for _ in range(n_samples):
            configs.append(self.sample())
        return configs


def create_default_sampler(
    n_actions: int = 3,
    random_seed: int = 42,
    distribution: Literal["uniform", "normal"] = "uniform",
    stability_method: StabilityMethod = "eigenvalues",
    g_range: tuple[float, float] = (0.0, 3.0),
    mu_range: tuple[float, float] = (-10.0, 0.0),
    Pi_diag_range: tuple[float, float] = (0.1, 3.0),
    Pi_offdiag_range: tuple[float, float] = (-3.0, 0.0),
    Gamma_diag_range: tuple[float, float] = (0.1, 3.0),
    Gamma_offdiag_range: tuple[float, float] = (-3.0, 3.0),
    v_range: tuple[float, float] = (-5.0, 5.0),
    m_range: tuple[float, float] = (-2.0, 2.0),
) -> ODEConfigSampler:
    """
    Create a default ODEConfigSampler with sensible parameter ranges.

    Args:
        n_actions: Number of actions (n)
        random_seed: Random seed for reproducibility
        distribution: Distribution type ('uniform' or 'normal')
        stability_method: Method to use for stability checking (StabilityMethod), default: "eigenvalues"
        g_range: Range for g parameter, default: (0.0, 3.0)
        mu_range: Range for mu parameter, default: (-10.0, 0.0)
        Pi_diag_range: Range for Pi diagonal parameter, default: (0.1, 3.0)
        Pi_offdiag_range: Range for Pi offdiag parameter, default: (-3.0, 0.0)
        Gamma_diag_range: Range for Gamma diagonal parameter, default: (0.1, 3.0)
        Gamma_offdiag_range: Range for Gamma offdiag parameter, default: (-3.0, 3.0)
        v_range: Range for v initial condition, default: (-5.0, 5.0)
        m_range: Range for m initial condition, default: (-2.0, 2.0)

    Returns:
        Configured ODEConfigSampler

    Example:
        >>> sampler = create_default_sampler(n_actions=3, random_seed=42)
        >>> config = sampler.sample()
    """
    n = n_actions

    param_min_dict = {
        "g": np.full(n, g_range[0], dtype=float),
        "mu": np.full(n, mu_range[0], dtype=float),
        "Pi": np.full((n, n), Pi_offdiag_range[0], dtype=float),
        "Gamma": np.full((n, n), Gamma_offdiag_range[0], dtype=float),
    }
    param_max_dict = {
        "g": np.full(n, g_range[1], dtype=float),
        "mu": np.full(n, mu_range[1], dtype=float),
        "Pi": np.full((n, n), Pi_offdiag_range[1], dtype=float),
        "Gamma": np.full((n, n), Gamma_offdiag_range[1], dtype=float),
    }

    for i in range(n):
        param_min_dict["Pi"][i, i] = Pi_diag_range[0]
        param_max_dict["Pi"][i, i] = Pi_diag_range[1]
        param_min_dict["Gamma"][i, i] = Gamma_diag_range[0]
        param_max_dict["Gamma"][i, i] = Gamma_diag_range[1]

    param_min = ParameterVector.from_dict(param_min_dict)
    param_max = ParameterVector.from_dict(param_max_dict)

    ic_min_dict = {
        "v": np.full(n, v_range[0], dtype=float),
        "m": np.full(n, m_range[0], dtype=float),
    }
    ic_max_dict = {
        "v": np.full(n, v_range[1], dtype=float),
        "m": np.full(n, m_range[1], dtype=float),
    }

    ic_min = ICVector.from_dict(ic_min_dict)
    ic_max = ICVector.from_dict(ic_max_dict)

    return ODEConfigSampler(
        random_seed=random_seed,
        distribution=distribution,
        stability_method=stability_method,
        parameter_ranges=(param_min, param_max),
        ic_range=(ic_min, ic_max),
    )
