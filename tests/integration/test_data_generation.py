"""
Integration tests for data generation pipeline.
"""

import numpy as np

from preference_dynamics.schemas import (
    ICVector,
    ParameterVector,
    SolverConfig,
)
from preference_dynamics.solver import (
    create_default_sampler,
    generate_batch,
)


class TestEndToEndDataGeneration:
    """Integration tests for end-to-end data generation."""

    def test_complete_pipeline_n1(self) -> None:
        """Test complete data generation pipeline for n=1."""

        sampler = create_default_sampler(n_actions=1, random_seed=47)
        solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=10)
        samples = generate_batch(
            n_samples=2, sampler=sampler, solver_config=solver_config, show_progress=False
        )

        assert len(samples) == 2

        for sample in samples:
            assert hasattr(sample, "config")
            assert hasattr(sample.config, "ode")
            assert hasattr(sample.config, "solver")
            assert sample.config.solver == solver_config

            assert sample.n_actions == 1
            assert sample.time_series.shape == (2, 10)
            assert sample.parameters.values.shape == (4,)
            assert sample.initial_conditions.values.shape == (2,)

    def test_complete_pipeline_n3(self) -> None:
        """Test complete data generation pipeline for n=3."""
        sampler = create_default_sampler(n_actions=3, random_seed=74)
        solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=10)

        samples = generate_batch(
            n_samples=2, sampler=sampler, solver_config=solver_config, show_progress=False
        )

        assert len(samples) == 2

        for sample in samples:
            assert sample.n_actions == 3
            assert sample.time_series.shape == (6, 10)
            assert sample.parameters.values.shape == (24,)
            assert sample.initial_conditions.values.shape == (6,)


class TestReproducibility:
    """Integration tests for reproducibility."""

    def test_same_seed_produces_same_results(self) -> None:
        """Test that same random seed produces identical results."""
        solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=10)

        sampler1 = create_default_sampler(n_actions=2, random_seed=47)
        samples1 = generate_batch(
            n_samples=2, sampler=sampler1, solver_config=solver_config, show_progress=False
        )
        sampler2 = create_default_sampler(n_actions=2, random_seed=47)
        samples2 = generate_batch(
            n_samples=2, sampler=sampler2, solver_config=solver_config, show_progress=False
        )

        assert len(samples1) == len(samples2)

        for s1, s2 in zip(samples1, samples2, strict=True):
            np.testing.assert_allclose(s1.parameters.values, s2.parameters.values, rtol=1e-10)
            np.testing.assert_allclose(
                s1.initial_conditions.values, s2.initial_conditions.values, rtol=1e-10
            )

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=10)

        sampler1 = create_default_sampler(n_actions=2, random_seed=47)
        samples1 = generate_batch(
            n_samples=2, sampler=sampler1, solver_config=solver_config, show_progress=False
        )

        sampler2 = create_default_sampler(n_actions=2, random_seed=74)
        samples2 = generate_batch(
            n_samples=2, sampler=sampler2, solver_config=solver_config, show_progress=False
        )

        for s1, s2 in zip(samples1, samples2, strict=True):
            assert not np.allclose(s1.parameters.values, s2.parameters.values)

    # TODO: move to sampler unit tests
    def test_parameter_ranges_applied(self) -> None:
        """Test that parameter ranges are being used."""
        sampler = create_default_sampler(
            n_actions=3,
            mu_range=(0.0, 1.0),
            Pi_diag_range=(4.0, 5.0),
            Gamma_diag_range=(4.0, 5.0),
            random_seed=789,
        )
        configs = sampler.sample_batch(n_samples=10)

        for config in configs:
            params = config.parameters

            mu = params.mu
            assert np.all(mu >= 0.0)
            assert np.all(mu <= 1.0)

            Pi = params.Pi
            for i in range(3):
                assert Pi[i, i] >= 4.0
                assert Pi[i, i] <= 5.0

            Gamma = params.Gamma
            for i in range(3):
                assert Gamma[i, i] >= 4.0
                assert Gamma[i, i] <= 5.0


class TestPropertyBasedAccess:
    """Integration tests for property-based access."""

    # TODO: move to schemas unit tests, and check if duplicates exist
    # TODO: reduce to n=2
    def test_parameter_vector_properties(self) -> None:
        """Test that ParameterVector properties work (.g, .mu, .Pi, .Gamma)."""
        params = ParameterVector(
            values=np.array(
                [
                    1.0,
                    1.5,
                    2.0,  # g
                    -1.0,
                    -0.5,
                    -0.5,  # μ
                    2.0,
                    -0.1,
                    -0.2,
                    -0.1,
                    1.5,
                    -0.3,
                    -0.2,
                    -0.3,
                    1.0,  # Π
                    1.0,
                    -0.05,
                    0.1,
                    -0.1,
                    2.0,
                    -0.15,
                    -0.05,
                    -0.2,
                    1.5,  # Γ
                ]
            )
        )

        # Test property access
        assert params.g.shape == (3,)
        assert params.mu.shape == (3,)
        assert params.Pi.shape == (3, 3)
        assert params.Gamma.shape == (3, 3)

        # Test values
        np.testing.assert_array_equal(params.g, np.array([1.0, 1.5, 2.0]))
        np.testing.assert_array_equal(params.mu, np.array([-1.0, -0.5, -0.5]))
        np.testing.assert_array_equal(
            params.Pi,
            np.array([[2.0, -0.1, -0.2], [-0.1, 1.5, -0.3], [-0.2, -0.3, 1.0]]),
        )
        np.testing.assert_array_equal(
            params.Gamma, np.array([[1.0, -0.05, 0.1], [-0.1, 2.0, -0.15], [-0.05, -0.2, 1.5]])
        )

    # TODO: move to schemas unit tests, and check if duplicates exist
    # TODO: reduce to n=2
    def test_ic_vector_properties(self) -> None:
        """Test that ICVector properties work (.v, .m)."""
        ic = ICVector(values=np.array([1.0, 2.0, 3.0, 0.5, 1.0, 1.5]))

        # Test property access
        assert ic.v.shape == (3,)
        assert ic.m.shape == (3,)

        # Test values
        np.testing.assert_array_equal(ic.v, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(ic.m, np.array([0.5, 1.0, 1.5]))

    def test_end_to_end_with_property_access(self) -> None:
        """Test complete pipeline using only property access."""
        sampler = create_default_sampler(n_actions=2, random_seed=444)
        solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=10)
        samples = generate_batch(
            n_samples=1, sampler=sampler, solver_config=solver_config, show_progress=False
        )

        sample = samples[0]

        g = sample.parameters.g
        mu = sample.parameters.mu
        Pi = sample.parameters.Pi
        Gamma = sample.parameters.Gamma

        v0 = sample.initial_conditions.v
        m0 = sample.initial_conditions.m

        desires = sample.desires
        efforts = sample.efforts

        assert g.shape == (2,)
        assert mu.shape == (2,)
        assert Pi.shape == (2, 2)
        assert Gamma.shape == (2, 2)
        assert v0.shape == (2,)
        assert m0.shape == (2,)
        assert desires.shape == (2, 10)
        assert efforts.shape == (2, 10)
        np.testing.assert_array_equal(desires, sample.time_series[0:2, :])
        np.testing.assert_array_equal(efforts, sample.time_series[2:4, :])

        assert np.all(efforts >= -1e-10), "Efforts should be non-negative"
        # desires is (2, 100), mu is (2,), need to broadcast along time dimension
        assert np.all(desires >= mu[:, None] - 1e-10), "Desires should be >= μ"
