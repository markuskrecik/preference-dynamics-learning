"""
Unit tests for Pydantic schemas.
"""

import uuid

import numpy as np
import pytest

from preference_dynamics.schemas import (
    ICVector,
    ODEConfig,
    ODESolverConfig,
    ParameterVector,
    SolverConfig,
    TimeSeriesSample,
    TrainingHistory,
)

# ============================================================================
# T025: Tests for ParameterVector validation
# ============================================================================


class TestParameterVector:
    """Test suite for ParameterVector Pydantic schema."""

    def test_valid_n1_parameters(self) -> None:
        """Test valid parameter vector for n=1."""
        params = ParameterVector(
            n_actions=1,
            values=np.array([1.0, 0.5, 2.0, 1.5]),  # g, mu, Pi, Gamma (4 params total)
        )
        assert params.n_actions == 1
        assert params.values.shape == (4,)

    def test_valid_n2_parameters(self) -> None:
        """Test valid parameter vector for n=2."""
        params = ParameterVector(
            n_actions=2,
            values=np.array(
                [
                    1.0,
                    1.5,  # g (2)
                    0.0,
                    0.5,  # mu (2)
                    2.0,
                    0.1,
                    -0.1,
                    1.5,  # Pi (4)
                    1.0,
                    0.05,
                    0.1,
                    2.0,  # Gamma (4)
                ]
            ),
        )
        assert params.n_actions == 2
        assert params.values.shape == (12,)

    def test_valid_n3_parameters(self) -> None:
        """Test valid parameter vector for n=3."""
        params = ParameterVector(
            n_actions=3,
            values=np.array(
                [
                    1.0,
                    1.5,
                    2.0,  # g (3)
                    0.0,
                    0.5,
                    -0.5,  # mu (3)
                    2.0,
                    0.1,
                    0.2,
                    -0.1,
                    1.5,
                    0.3,
                    0.2,
                    -0.2,
                    1.0,  # Pi (9)
                    1.0,
                    0.05,
                    -0.1,
                    0.1,
                    2.0,
                    0.15,
                    -0.05,
                    0.2,
                    1.5,  # Gamma (9)
                ]
            ),
        )
        assert params.n_actions == 3
        assert params.values.shape == (24,)

    def test_invalid_shape(self) -> None:
        """Test that wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="must have shape"):
            ParameterVector(
                n_actions=3,
                values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Wrong size (5 instead of 24)
            )

    def test_negative_diagonal_pi(self) -> None:
        """Test that negative Π diagonal raises ValueError."""
        params_array = np.ones(24)
        params_array[6] = -1.0  # Negative Π[0,0]

        with pytest.raises(ValueError, match="Diagonal element.*must be > 0"):
            ParameterVector(n_actions=3, values=params_array)

    def test_negative_diagonal_gamma(self) -> None:
        """Test that negative Γ diagonal raises ValueError."""
        params_array = np.ones(24)
        params_array[15] = -1.0  # Negative Γ[0,0]

        with pytest.raises(ValueError, match="Diagonal element.*must be > 0"):
            ParameterVector(n_actions=3, values=params_array)

    def test_zero_diagonal_rejected(self) -> None:
        """Test that zero diagonals are rejected."""
        params_array = np.ones(24)
        params_array[10] = 0.0  # Zero Π[1,1]

        with pytest.raises(ValueError, match="Diagonal element.*must be > 0"):
            ParameterVector(n_actions=3, values=params_array)

    def test_get_g(self) -> None:
        """Test g property extracts drive vector correctly."""
        params = ParameterVector(
            n_actions=3,
            values=np.array(
                [
                    1.0,
                    2.0,
                    3.0,  # g
                    0.0,
                    0.0,
                    0.0,  # mu
                    *[1.0] * 18,  # Pi and Gamma
                ]
            ),
        )
        g = params.g
        np.testing.assert_array_equal(g, [1.0, 2.0, 3.0])

    def test_get_mu(self) -> None:
        """Test mu property extracts thresholds correctly."""
        params = ParameterVector(
            n_actions=3,
            values=np.array(
                [
                    1.0,
                    1.0,
                    1.0,  # g
                    4.0,
                    5.0,
                    6.0,  # mu
                    *[1.0] * 18,  # Pi and Gamma
                ]
            ),
        )
        mu = params.mu
        np.testing.assert_array_equal(mu, [4.0, 5.0, 6.0])

    def test_get_Pi(self) -> None:
        """Test Pi property returns correct matrix shape."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        Pi = params.Pi
        assert Pi.shape == (3, 3)

    def test_get_Gamma(self) -> None:
        """Test Gamma property returns correct matrix shape."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        Gamma = params.Gamma
        assert Gamma.shape == (3, 3)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test that to_dict() and from_dict() are inverse operations."""
        original = ParameterVector(n_actions=2, values=np.arange(12, dtype=float))
        d = original.to_dict()
        reconstructed = ParameterVector.from_dict(d)

        np.testing.assert_array_equal(original.values, reconstructed.values)
        assert original.n_actions == reconstructed.n_actions


# ============================================================================
# T026: Tests for ODEConfig validation
# ============================================================================


class TestODEConfig:
    """Test suite for ODEConfig Pydantic schema."""

    def test_valid_ode_config_n2(self) -> None:
        """Test valid ODE configuration for n=2."""
        params = ParameterVector(n_actions=2, values=np.ones(12))
        ic = ICVector(values=np.random.randn(4))

        config = ODEConfig(
            n_actions=2,
            parameters=params,
            initial_conditions=ic,
        )

        assert config.n_actions == 2
        assert config.parameters.n_actions == 2
        assert config.initial_conditions.n_actions == 2

    def test_parameter_n_actions_mismatch(self) -> None:
        """Test that mismatched n_actions raises ValueError."""
        params = ParameterVector(n_actions=2, values=np.ones(12))
        ic = ICVector(values=np.random.randn(6))

        with pytest.raises(ValueError, match="Parameters n_actions.*must match"):
            ODEConfig(
                n_actions=3,  # Mismatch!
                parameters=params,
                initial_conditions=ic,
            )

    def test_wrong_initial_conditions_shape(self) -> None:
        """Test that wrong IC shape raises ValueError."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic_values = np.random.randn(4)  # Should be 6 for n=3

        with pytest.raises(ValueError, match="Initial conditions.*must have shape"):
            ic = ICVector(values=ic_values, n_actions=3)  # This should fail
            ODEConfig(
                n_actions=3,
                parameters=params,
                initial_conditions=ic,
            )

    def test_invalid_time_span(self) -> None:
        """Test that t_end <= t_start raises ValueError."""
        with pytest.raises(ValueError, match="t_end must be > t_start"):
            SolverConfig(
                time_span=(100.0, 50.0),  # Invalid!
                n_time_points=1000,
            )

    def test_different_solver_methods(self) -> None:
        """Test that different solver methods are accepted."""
        for method in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]:
            config = SolverConfig(
                time_span=(0.0, 100.0),
                n_time_points=1000,
                solver_method=method,
            )
            assert config.solver_method == method


# ============================================================================
# T027: Tests for TimeSeriesSample validation
# ============================================================================


class TestTimeSeriesSample:
    """Test suite for TimeSeriesSample Pydantic schema."""

    def test_valid_sample_n3(self) -> None:
        """Test valid time series sample for n=3."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3:6, :] = np.abs(time_series[3:6, :])  # Ensure efforts >= 0

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        sample = TimeSeriesSample(
            sample_id=str(uuid.uuid4()),
            time_series=time_series,
            time_points=np.linspace(0, 100, 1000),
            config=config,
        )

        assert sample.n_actions == 3
        assert sample.sequence_length == 1000
        assert sample.time_series.shape == (6, 1000)

    def test_wrong_timeseries_shape(self) -> None:
        """Test that wrong timeseries shape raises ValueError."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(4, 1000)  # Should be (6, 1000) for n=3

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        with pytest.raises(ValueError, match="Time series.*must have shape"):
            TimeSeriesSample(
                sample_id=str(uuid.uuid4()),
                time_series=time_series,
                time_points=np.linspace(0, 100, 1000),
                config=config,
            )

    @pytest.mark.skip(reason="Positive effort enforcement not compatible with normalization.")
    def test_negative_efforts_rejected(self) -> None:
        """Test that negative efforts raise ValueError."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3, 500] = -1.0  # Negative effort!

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        with pytest.raises(ValueError, match="Efforts must be >= 0"):
            TimeSeriesSample(
                sample_id=str(uuid.uuid4()),
                time_series=time_series,
                time_points=np.linspace(0, 100, 1000),
                config=config,
            )

    def test_small_negative_efforts_tolerated(self) -> None:
        """Test that small numerical errors in efforts are tolerated."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3:6, :] = np.abs(time_series[3:6, :])
        time_series[3, 500] = -1e-11  # Small numerical error, should be OK

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        sample = TimeSeriesSample(
            sample_id=str(uuid.uuid4()),
            time_series=time_series,
            time_points=np.linspace(0, 100, 1000),
            config=config,
        )

        assert sample.n_actions == 3

    def test_get_desires_method(self) -> None:
        """Test desires property extracts correct channels."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3:6, :] = np.abs(time_series[3:6, :])

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        sample = TimeSeriesSample(
            sample_id=str(uuid.uuid4()),
            time_series=time_series,
            time_points=np.linspace(0, 100, 1000),
            config=config,
        )

        desires = sample.desires
        assert desires.shape == (3, 1000)
        np.testing.assert_array_equal(desires, time_series[0:3, :])

    def test_get_efforts_method(self) -> None:
        """Test efforts property extracts correct channels."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3:6, :] = np.abs(time_series[3:6, :])

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        sample = TimeSeriesSample(
            sample_id=str(uuid.uuid4()),
            time_series=time_series,
            time_points=np.linspace(0, 100, 1000),
            config=config,
        )

        efforts = sample.efforts
        assert efforts.shape == (3, 1000)
        np.testing.assert_array_equal(efforts, time_series[3:6, :])

    def test_parameter_n_actions_mismatch(self) -> None:
        """Test that mismatched parameter n_actions raises ValueError."""
        params = ParameterVector(n_actions=2, values=np.ones(12))
        ic = ICVector(values=np.random.randn(4))
        time_series = np.random.randn(1000, 6)
        time_series[:, 3:6] = np.abs(time_series[:, 3:6])

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        with pytest.raises(ValueError, match="Time series.*must have shape"):
            TimeSeriesSample(
                sample_id=str(uuid.uuid4()),
                time_series=time_series,  # Wrong shape for n=2
                time_points=np.linspace(0, 100, 1000),
                config=config,
            )

    def test_time_points_shape_mismatch(self) -> None:
        """Test that time_points shape must match T."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3:6, :] = np.abs(time_series[3:6, :])

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        with pytest.raises(ValueError, match="Time points must have shape"):
            TimeSeriesSample(
                sample_id=str(uuid.uuid4()),
                time_series=time_series,
                time_points=np.linspace(0, 100, 500),  # Wrong length!
                config=config,
            )

    def test_sample_with_metadata(self) -> None:
        """Test that metadata is stored correctly."""
        params = ParameterVector(n_actions=3, values=np.ones(24))
        ic = ICVector(values=np.random.randn(6))
        time_series = np.random.randn(6, 1000)
        time_series[3:6, :] = np.abs(time_series[3:6, :])

        metadata = {"solver_method": "RK45", "generation_time": 1.23, "custom_field": "value"}

        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=42)
        solver_config = SolverConfig(time_span=(0.0, 100.0), n_time_points=1000)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)

        sample = TimeSeriesSample(
            sample_id=str(uuid.uuid4()),
            time_series=time_series,
            time_points=np.linspace(0, 100, 1000),
            config=config,
            metadata=metadata,
        )

        assert sample.metadata["solver_method"] == "RK45"
        assert sample.metadata["generation_time"] == 1.23
        assert sample.metadata["custom_field"] == "value"


def test_training_history_validation_errors() -> None:
    with pytest.raises(ValueError, match="val_loss length"):
        TrainingHistory(
            train_loss=[1.0, 0.9],
            val_loss=[0.9],
            epoch_times=[1.0, 1.0],
            best_epoch=1,
            best_loss=0.9,
            total_epochs=1,
            converged=False,
        )
    with pytest.raises(ValueError, match="epoch_times values"):
        TrainingHistory(
            train_loss=[1.0],
            val_loss=[1.0],
            epoch_times=[-1.0],
            best_epoch=0,
            best_loss=1.0,
            total_epochs=1,
            converged=False,
        )
