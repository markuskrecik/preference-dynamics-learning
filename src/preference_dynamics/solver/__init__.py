"""
ODE solver and data generation for preference dynamics.

This module provides:
- preference_dynamics_rhs: Right-hand side function for ODE system
- solve_ODE: Solve ODE and generate time series sample
- generate_batch: Generate multiple samples with stability filtering
- validate_solution: Validate time series samples
- validate_constraints: Validate constraint satisfaction
- ODEConfigSampler: Generate random valid ODE configs
- check_stability: Check stability criteria for ODE config
- Constraint validation utilities

All functions support variable dimensions (n=1,2,3,... actions).
"""

from preference_dynamics.solver.batch import generate_batch
from preference_dynamics.solver.integrator import (
    solve_ODE,
)
from preference_dynamics.solver.sampler import (
    ODEConfigSampler,
    create_default_sampler,
)

__all__ = [
    "solve_ODE",
    "generate_batch",
    "ODEConfigSampler",
    "create_default_sampler",
]
