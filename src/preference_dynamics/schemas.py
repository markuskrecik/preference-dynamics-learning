"""
Pydantic schemas for preference dynamics parameter learning.

All schemas support variable dimensions (n actions) with automatic inference.
Dimension scaling: n actions → 2n state dims → 2n+2n² parameters.
"""

import uuid
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import numpy as np
from numpydantic import NDArray
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from torch import nn

from preference_dynamics.utils import get_diagonal_indices

# ============================================================================
# Parameter Configuration
# ============================================================================


class ParameterVector(BaseModel):
    """
    Parameter vector for preference dynamics (supports variable n actions with auto-inference).

    Structure:
        - g [0:n]: Drive vector (n params)
        - μ [n:2n]: Desire thresholds (n params)
        - Π [2n:2n+n²]: Coupling matrix Π, row-major (n² params)
        - Γ [2n+n²:2n+2n²]: Coupling matrix Γ, row-major (n² params)

    Total parameters: 2n + 2n²

    Constraints:
        - All Π diagonal elements > 0
        - All Γ diagonal elements > 0
    """

    values: NDArray[Literal["4-*"], float] = Field(
        ..., description="Parameter vector (2n + 2n²,), minimum 4 elements"
    )
    n_actions: int | None = Field(
        default=None, description="Number of actions (n), auto-inferred if None", ge=1
    )

    def __repr__(self) -> str:
        return f"ParameterVector.from_dict({self.to_dict()})"

    @model_validator(mode="after")
    def infer_n_actions(self) -> Self:
        """Infer n_actions from values shape if not provided."""
        if self.n_actions is None:
            # Solve: 2n + 2n² = len(values) for n
            # => 2n² + 2n - len = 0
            # => n = (-2 + sqrt(4 + 8*len)) / 4 = (-1 + sqrt(1 + 2*len)) / 2
            length = self.values.shape[0]
            n = (np.sqrt(1 + 2 * length) - 1) / 2
            if not n.is_integer():
                raise ValueError(
                    f"Cannot infer valid n_actions from length {length}. "
                    f"Length must satisfy 2n + 2n² for integer n ≥ 1"
                )
            self.n_actions = int(n)
        return self

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate shape matches n_actions."""
        n: int = self.n_actions  # type: ignore
        expected_length = 2 * n + 2 * n**2

        if self.values.shape != (expected_length,):
            raise ValueError(
                f"Parameter vector for n={n} must have shape ({expected_length},), "
                f"got {self.values.shape}"
            )
        return self

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        """Validate diagonal positivity constraints."""
        n: int = self.n_actions  # type: ignore
        diagonal_indices = get_diagonal_indices(n)
        for idx in diagonal_indices:
            if self.values[idx] <= 0:
                raise ValueError(
                    f"Diagonal element at index {idx} must be > 0, got {self.values[idx]}"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with named parameters."""
        n: int = self.n_actions  # type: ignore
        return {
            "n_actions": n,
            "g": self.values[0:n].tolist(),
            "mu": self.values[n : 2 * n].tolist(),
            "Pi": self.values[2 * n : 2 * n + n**2].reshape(n, n).tolist(),
            "Gamma": self.values[2 * n + n**2 : 2 * n + 2 * n**2].reshape(n, n).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create from dictionary."""
        g = np.array(d["g"])
        mu = np.array(d["mu"])
        Pi = np.array(d["Pi"]).flatten()
        Gamma = np.array(d["Gamma"]).flatten()

        values = np.concatenate([g, mu, Pi, Gamma])
        return cls(values=values, n_actions=len(g))

    @property
    def g(self) -> NDArray[Literal["*"], float]:
        """Get drive vector (n,)."""
        return self.values[0 : self.n_actions]

    @property
    def mu(self) -> NDArray[Literal["*"], float]:
        """Get desire thresholds (n,)."""
        n: int = self.n_actions  # type: ignore
        return self.values[n : 2 * n]

    @property
    def Pi(self) -> NDArray[Literal["*, *"], float]:
        """Get Π coupling matrix (n, n)."""
        n: int = self.n_actions  # type: ignore
        return self.values[2 * n : 2 * n + n**2].reshape(n, n)

    @property
    def Gamma(self) -> NDArray[Literal["*, *"], float]:
        """Get Γ coupling matrix (n, n)."""
        n: int = self.n_actions  # type: ignore
        return self.values[2 * n + n**2 : 2 * n + 2 * n**2].reshape(n, n)


# ============================================================================
# Initial Conditions Vector
# ============================================================================


class ICVector(BaseModel):
    """
    Initial conditions vector for preference dynamics (auto-inference supported).

    Structure:
        - v [0:n]: Initial desire potentials (n values)
        - m [n:2n]: Initial effort potentials (n values)

    Total: 2n values
    """

    values: NDArray[Literal["2-*"], float] = Field(
        ..., description="Initial conditions vector (2n,), minimum 2 elements"
    )
    n_actions: int | None = Field(
        default=None, description="Number of actions (n), auto-inferred if None", ge=1
    )

    def __repr__(self) -> str:
        return f"ICVector.from_dict({self.to_dict()})"

    @model_validator(mode="after")
    def infer_n_actions(self) -> Self:
        """Infer n_actions from values shape if not provided."""
        if self.n_actions is None:
            length = self.values.shape[0]
            if length % 2 != 0:
                raise ValueError(
                    f"Cannot infer n_actions from odd length {length}. Length must be even (2n)"
                )
            self.n_actions = length // 2
        return self

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate shape matches n_actions."""
        n: int = self.n_actions  # type: ignore
        expected_length = 2 * n

        if self.values.shape != (expected_length,):
            raise ValueError(
                f"Initial conditions for {n=} must have shape ({expected_length},), "
                f"got {self.values.shape}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        n: int = self.n_actions  # type: ignore
        return {
            "n_actions": n,
            "v": self.values[0:n].tolist(),
            "m": self.values[n : 2 * n].tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create from dictionary."""
        v = np.array(d["v"])
        m = np.array(d["m"])
        values = np.concatenate([v, m])
        return cls(values=values, n_actions=len(v))

    @property
    def v(self) -> NDArray[Literal["*"], float]:
        """Get initial desire potentials (n,)."""
        return self.values[0 : self.n_actions]

    @property
    def m(self) -> NDArray[Literal["*"], float]:
        """Get initial effort potentials (n,)."""
        n: int = self.n_actions  # type: ignore
        return self.values[n : 2 * n]


# ============================================================================
# ODE Configuration
# ============================================================================


class ODEConfig(BaseModel):
    """
    ODE system configuration (parameters and initial conditions only).

    Separates mathematical system definition from numerical solver settings.
    """

    parameters: ParameterVector = Field(..., description="Parameter vector (2n + 2n² dimensional)")

    initial_conditions: ICVector = Field(
        ..., description="Initial state [v_1, ..., v_n, m_1, ..., m_n] (2n dimensional)"
    )

    n_actions: int | None = Field(
        default=None, description="Number of actions (n), auto-inferred if None", ge=1
    )

    @model_validator(mode="after")
    def infer_n_actions(self) -> Self:
        """Infer n_actions from initial_conditions if not provided."""
        if self.n_actions is None:
            self.n_actions = self.initial_conditions.n_actions
        return self

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate that dimensions are consistent with n_actions."""
        n: int = self.n_actions  # type: ignore
        if self.parameters.n_actions != n:
            raise ValueError(
                f"Parameters n_actions ({self.parameters.n_actions}) must match "
                f"ODEConfig n_actions ({n})"
            )

        if self.initial_conditions.n_actions != n:
            raise ValueError(
                f"Initial conditions n_actions ({self.initial_conditions.n_actions}) must match "
                f"ODEConfig n_actions ({n})"
            )

        return self


class StabilityReport(BaseModel):
    """
    Report of stability analysis for an ODE configuration.
    """

    is_stable: bool = Field(
        ..., description="Whether the configuration satisfies stability criteria"
    )
    actions: list[int] | None = Field(default=None, description="Performed equilibrium actions")
    desired_actions: list[int] | None = Field(
        default=None, description="Desired equilibrium actions"
    )
    eigenvalues: NDArray | None = Field(default=None, description="Eigenvalues of the Jacobian")
    eigenvectors: NDArray | None = Field(
        default=None, description="Eigenvectors (columns) of the Jacobian"
    )


# ============================================================================
# Solver Configuration
# ============================================================================


class SolverConfig(BaseModel):
    """
    Numerical solver configuration for ODE integration.

    Separates solver settings from ODE system definition for better modularity.
    """

    time_span: tuple[float, float] = Field(
        ..., description="Integration time span (t_start, t_end)"
    )

    n_time_points: int = Field(..., ge=2, description="Number of time steps for evaluation (T)")

    solver_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = Field(
        default="RK45", description="scipy.integrate.solve_ivp method"
    )

    rtol: float = Field(default=1e-6, ge=1e-12, description="Relative tolerance for ODE solver")

    atol: float = Field(default=1e-9, ge=1e-15, description="Absolute tolerance for ODE solver")

    events: list[Callable] | None = Field(  # type: ignore
        default=None,
        description="Event functions for solve_ivp (e.g., detect m=0 crossings). "
        "Will be implemented later.",
    )

    @model_validator(mode="after")
    def validate_time_span(self) -> Self:
        """Validate time span is valid."""
        if self.time_span[1] <= self.time_span[0]:
            raise ValueError(f"t_end must be > t_start, got {self.time_span}")
        return self


# ============================================================================
# Combined ODE Solver Configuration
# ============================================================================


class ODESolverConfig(BaseModel):
    """
    Complete configuration combining ODE system and solver settings.

    This is the top-level configuration stored in TimeSeriesSample.
    """

    ode: ODEConfig = Field(..., description="ODE system configuration (parameters, ICs)")

    solver: SolverConfig = Field(..., description="Numerical solver configuration")


# ============================================================================
# Time Series Sample
# ============================================================================


class SampleStatistics(BaseModel):
    """Statistics for a dataset split."""

    method: str = Field(..., description="Method used to calculate statistics.")
    means: NDArray[Literal["2-*"], float] = Field(
        ..., description="Sample means, calculated by specified method."
    )
    stds: NDArray[Literal["2-*"], float] = Field(
        ..., description="Sample standard deviations, calculated by specified method."
    )


class TimeSeriesSample(BaseModel):
    """
    Single time series sample with complete configuration.

    Stores both the time series observations and the complete configuration
    used to generate it (parameters, ICs, solver settings).
    """

    sample_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID), auto-generated",
    )

    time_series: NDArray[Literal["2-*, *"], float] = Field(
        ..., description="Observed trajectory [u, a] with shape (2n, T)"
    )

    time_points: NDArray[Literal["*"], float] = Field(
        ..., description="Actual time values with shape (T,)"
    )

    config: ODESolverConfig = Field(
        ...,
        description="Complete ODE solver configuration used to generate this sample",
    )

    statistics: SampleStatistics | None = Field(
        default=None, description="Statistics of the sample"
    )

    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted features of the sample",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the solution (convergence, generation time, etc.)",
    )

    # model_config = {"json_encoders": {np.ndarray: lambda a: a.tolist()}}

    @field_validator("time_series", "time_points", mode="before")
    @classmethod
    def to_array(
        cls, v: Sequence[float] | Sequence[Sequence[float]]
    ) -> NDArray[Literal["*, ..."], float]:
        return np.asarray(v, dtype=float)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate dimensions match configuration."""
        n: int = self.config.ode.n_actions  # type: ignore
        T = self.config.solver.n_time_points

        if self.time_series.shape != (2 * n, T):
            raise ValueError(
                f"Time series must have shape {(2 * n, T)=}, got {self.time_series.shape}"
            )

        if self.time_points.shape != (T,):
            raise ValueError(f"Time points must have shape {(T,)=}, got {self.time_points.shape}")

        return self

    # @model_validator(mode="after")
    # def validate_constraints(self) -> Self:
    #     """Validate effort non-negativity constraint."""
    #     n: int = self.config.ode.n_actions  # type: ignore

    #     # Validate effort non-negativity (channels n to 2n-1)
    #     efforts = self.time_series[n : 2 * n, :]
    #     if np.any(efforts < -1e-10):  # Small tolerance for numerical errors
    #         min_effort = efforts.min()
    #         raise ValueError(f"Efforts must be >= 0, found min effort = {min_effort}")

    #     return self

    @property
    def n_actions(self) -> int:
        """Get number of actions from config."""
        return self.config.ode.n_actions  # type: ignore

    @property
    def sequence_length(self) -> int:
        """Get time series length."""
        length: int = self.time_series.shape[1]
        return length

    @property
    def desires(self) -> NDArray[Literal["*, *"], float]:
        """Get desire channels (u_1, ..., u_n)."""
        n: int = self.n_actions
        return self.time_series[0:n, :]

    @property
    def efforts(self) -> NDArray[Literal["*, *"], float]:
        """Get effort channels (a_1, ..., a_n)."""
        n: int = self.n_actions
        return self.time_series[n : 2 * n, :]

    @property
    def parameters(self) -> ParameterVector:
        """Get ground truth parameters from config."""
        return self.config.ode.parameters

    @property
    def initial_conditions(self) -> ICVector:
        """Get initial conditions from config."""
        return self.config.ode.initial_conditions


# ============================================================================
# Trainer Configuration
# ============================================================================


class PINNLossConfig(BaseModel):
    """
    Configuration for PINN loss computation.

    Fields:
        data_weight: Weight for data fitting loss
        physics_weight: Weight for physics residual loss
        supervised_weight: Weight for supervised parameter/IC loss (optional)
    """

    data_weight: float = Field(..., ge=0.0, description="Weight for data fitting loss")
    physics_weight: float = Field(..., ge=0.0, description="Weight for physics residual loss")
    supervised_weight: float = Field(
        default=0.0, ge=0.0, description="Weight for supervised parameter/IC loss"
    )
    # collocation: Literal["observed_times", "uniform", "random"] = Field(
    #     default="observed_times", description="Collocation strategy"
    # )
    # non_smooth_observables: bool = Field(
    #     default=False, description="Whether to use exact piecewise observation mapping"
    # )
    # smoothness_temperature: float | None = Field(
    #     default=None, ge=0.0, description="Temperature parameter for smooth approximations"
    # )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """Validate that at least one weight is positive."""
        if self.data_weight == 0.0 and self.physics_weight == 0.0:
            raise ValueError("At least one of data_weight or physics_weight must be > 0")
        return self


class TrainerConfig(BaseModel):
    """
    Configuration object for training hyperparameters.

    Fields:
        loss_function: Loss function identifier ("mse", "mae", "huber") or callable
        learning_rate: Learning rate (default: 0.001)
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience (None = disabled)
        weight_decay: L2 regularization (default: 1e-5)
        gradient_clip_norm: Gradient clipping norm (default: 1.0)
        device: Device ("cpu", "cuda", "mps", or None for auto-detect)
        seed: Random seed (default: 42)
        save_best: Only save best checkpoint (default: True)
    """

    loss_function: Literal["mse", "mae", "huber"] | nn.Module = Field(
        default="mse", description='Loss function identifier ("mse", "mae", "huber") or nn.Module'
    )
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    num_epochs: int = Field(..., gt=0, description="Number of training epochs")
    early_stopping_patience: int | None = Field(
        default=None, description="Early stopping patience (None = disabled)"
    )
    weight_decay: float = Field(default=1e-5, ge=0, description="L2 regularization")
    gradient_clip_norm: float = Field(default=1.0, gt=0, description="Gradient clipping norm")
    device: Literal["cpu", "cuda", "mps"] | None = Field(
        default=None, description='Device ("cpu", "cuda", "mps", or None for auto-detect)'
    )
    seed: int = Field(default=42, ge=0, description="Random seed")
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint directory")
    save_best: bool = Field(default=True, description="Only save best checkpoint")

    @model_validator(mode="after")
    def validate_early_stopping_patience(self) -> Self:
        """Validate early_stopping_patience is > 0 if not None."""
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be > 0 if not None, got {self.early_stopping_patience}"
            )
        return self

    model_config = {"arbitrary_types_allowed": True}


class TrainingHistory(BaseModel):
    """
    Record of training run results.

    Fields:
        train_loss: Training loss per epoch
        val_loss: Validation loss per epoch
        epoch_times: Epoch execution time (seconds) per epoch
        best_epoch: Epoch with best validation loss
        converged: Whether training converged (early stopping triggered)
    """

    train_loss: list[float] = Field(default_factory=list, description="Training loss per epoch")
    val_loss: list[float] = Field(default_factory=list, description="Validation loss per epoch")
    epoch_times: list[float] = Field(
        default_factory=list, description="Epoch execution time (seconds) per epoch"
    )
    best_epoch: int = Field(
        ...,
        ge=0,
        description="Epoch with best loss",
    )
    best_loss: float = Field(default=float("inf"), description="Best loss seen so far")
    total_epochs: int = Field(default=0, ge=0, description="Total number of epochs")
    converged: bool = Field(
        default=False, description="Whether training converged (early stopping triggered)"
    )

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        """Validate that all lists have same length, best_epoch is valid, and epoch_times >= 0."""
        n_epochs = len(self.train_loss)
        if len(self.val_loss) not in (n_epochs, 0):
            raise ValueError(
                f"val_loss length ({len(self.val_loss)}) must match train_loss length ({n_epochs}) or be 0"
            )
        if len(self.epoch_times) != n_epochs:
            raise ValueError(
                f"epoch_times length ({len(self.epoch_times)}) must match train_loss length ({n_epochs})"
            )
        if self.total_epochs != n_epochs:
            raise ValueError(
                f"total_epochs ({self.total_epochs}) must match train_loss length ({n_epochs})"
            )
        if any(t < 0 for t in self.epoch_times):
            raise ValueError("epoch_times values must be >= 0")
        return self


class RunnerConfig(BaseModel):
    """
    Configuration object for experiment-level settings.

    Fields:
        experiment_name: MLflow experiment name
        mlflow_tracking_uri: MLflow tracking URI (default: "sqlite:///mlruns.db")
        log_interval: Log metrics every N epochs (default: 1, must be >= 1)
        tags: MLflow experiment tags (default: {})
    """

    experiment_name: str = Field(..., min_length=1, description="MLflow experiment name")
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns.db", description="MLflow tracking URI"
    )
    optuna_storage: str | None = Field(
        default="sqlite:///optuna.db", description="Optuna storage URI"
    )
    log_interval: int = Field(default=1, ge=1, description="Log metrics every N epochs")
    tags: dict[str, str] = Field(default_factory=dict, description="MLflow experiment tags")
