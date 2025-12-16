"""
Unit tests for ODEConfigSampler stability and sampling behavior.
"""

import numpy as np

from preference_dynamics.solver.sampler import create_default_sampler


class TestSampler:
    # TODO: parametrize across n=1,2,3
    def test_sampler_respects_n_actions_and_ranges(self) -> None:
        sampler = create_default_sampler(n_actions=2, random_seed=0)

        config = sampler.sample()

        assert config.parameters.n_actions == 2
        assert config.initial_conditions.n_actions == 2
        assert config.parameters.values.shape == (12,)
        assert config.initial_conditions.values.shape == (4,)
        assert np.all(config.parameters.Gamma.diagonal() >= 0.1)
        assert np.all(config.parameters.Pi.diagonal() >= 0.1)

    def test_sample_batch_returns_configs(self) -> None:
        sampler = create_default_sampler(n_actions=1, random_seed=123)

        configs = sampler.sample_batch(n_samples=3)

        assert len(configs) == 3
        assert all(cfg.parameters.n_actions == 1 for cfg in configs)
        unique_params = {tuple(cfg.parameters.values.tolist()) for cfg in configs}
        assert len(unique_params) >= 2

    # TODO: test stability method
