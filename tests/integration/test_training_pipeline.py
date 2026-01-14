"""
Integration test for full training pipeline.

Tests verify that the complete training pipeline works end-to-end:
- DataManager setup
- Model creation
- Trainer initialization
- ExperimentRunner execution

TODO:
- any model, not just CNN1D
- n_jobs=2 for hyperparameter study
- save to optuna
- save to mlflow
"""

from pathlib import Path

import numpy as np
import pytest

from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.experiments.experiment import Experiment
from preference_dynamics.experiments.runner import ExperimentRunner
from preference_dynamics.models import CNN1DConfig
from preference_dynamics.schemas import (
    ICVector,
    ODEConfig,
    ODESolverConfig,
    ParameterVector,
    RunnerConfig,
    SolverConfig,
    TimeSeriesSample,
    TrainerConfig,
)
from preference_dynamics.utils import get_diagonal_indices


@pytest.mark.slow
def test_full_pipeline(
    cnn1d_config: CNN1DConfig, simple_trainer_config: TrainerConfig, tmp_path: Path
) -> None:
    """Test complete training pipeline from data loading to experiment execution."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    splits = (0.7, 0.15, 0.15)
    batch_size = 2
    min_samples = int(batch_size / min(splits) + 1)

    # TODO: use fixtures
    params = ParameterVector(n_actions=1, values=np.ones(4))
    for idx in get_diagonal_indices(1):
        params.values[idx] = np.abs(params.values[idx])
    ic = ICVector(n_actions=1, values=np.random.randn(2))
    ode_config = ODEConfig(parameters=params, initial_conditions=ic)
    solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=100)
    config = ODESolverConfig(ode=ode_config, solver=solver_config)

    samples = []
    for i in range(min_samples):
        time_series = np.random.randn(2, 100).astype(float)
        time_series[1:2, :] = np.abs(time_series[1:2, :])  # Ensure efforts >= 0
        samples.append(
            TimeSeriesSample(
                sample_id=f"sample_{i}",
                time_series=time_series,
                time_points=np.linspace(0, 10, 100).astype(float),
                config=config,
            )
        )

    data_config = DataConfig(
        data_dir=str(data_dir),
        load_if_exists=False,
        splits=splits,
        batch_size=batch_size,
        seed=42,
    )
    data_config.io_handler.save(samples, str(raw_dir / "train"))

    data_manager = DataManager(config=data_config)
    data_manager.setup()

    assert data_manager.train_dataloader is not None
    assert data_manager.val_dataloader is not None
    assert data_manager.test_dataloader is not None

    trainer_config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.001,
        num_epochs=2,
        seed=42,
    )
    runner_config = RunnerConfig(
        experiment_name="test_experiment",
        mlflow_tracking_uri="sqlite:///:memory:",
    )

    runner = ExperimentRunner(
        runner_config=runner_config,
        data_config=data_config,
        model_config=cnn1d_config,
        trainer_config=trainer_config,
    )

    experiment = runner.run(run_name="test_run")

    assert isinstance(experiment, Experiment)
    assert experiment.loss is not None
    assert experiment.loss >= 0.0


@pytest.mark.slow
def test_pipeline_with_hyperparameter_study(
    cnn1d_config: CNN1DConfig, simple_trainer_config: TrainerConfig, tmp_path: Path
) -> None:
    """Test complete pipeline with hyperparameter study."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    splits = (0.7, 0.15, 0.15)
    batch_size = 2
    min_samples = int(batch_size / min(splits) + 1)

    # TODO: use fixtures
    params = ParameterVector(n_actions=1, values=np.ones(4))
    for idx in get_diagonal_indices(1):
        params.values[idx] = np.abs(params.values[idx])
    ic = ICVector(n_actions=1, values=np.random.randn(2))
    ode_config = ODEConfig(parameters=params, initial_conditions=ic)
    solver_config = SolverConfig(time_span=(0.0, 10.0), n_time_points=100)
    config = ODESolverConfig(ode=ode_config, solver=solver_config)

    samples = []
    for i in range(min_samples):
        time_series = np.random.randn(2, 100).astype(float)
        time_series[1:2, :] = np.abs(time_series[1:2, :])  # Ensure efforts >= 0
        samples.append(
            TimeSeriesSample(
                sample_id=f"sample_{i}",
                time_series=time_series,
                time_points=np.linspace(0, 10, 100).astype(float),
                config=config,
            )
        )

    data_config = DataConfig(
        data_dir=str(data_dir),
        load_if_exists=False,
        splits=splits,
        batch_size=batch_size,
        seed=42,
    )
    data_config.io_handler.save(samples, str(raw_dir / "train"))

    data_manager = DataManager(config=data_config)
    data_manager.setup()

    assert data_manager.train_dataloader is not None
    assert data_manager.val_dataloader is not None
    assert data_manager.test_dataloader is not None

    trainer_config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.001,
        num_epochs=1,
        seed=42,
    )

    runner_config = RunnerConfig(
        experiment_name="test_experiment",
        mlflow_tracking_uri="sqlite:///:memory:",
        optuna_storage=None,
    )

    class CustomRunner(ExperimentRunner):
        def suggest_parameters(self, trial):
            self.data_config.batch_size = trial.suggest_int("batch_size", 1, 2, step=1)

    runner = CustomRunner(
        runner_config=runner_config,
        data_config=data_config,
        model_config=cnn1d_config,
        trainer_config=trainer_config,
    )

    study = runner.run_study(study_name="test_study", n_trials=2, n_jobs=1)

    assert study is not None
    assert len(study.trials) == 2
    assert study.best_trial is not None
    assert study.best_trial.value is not None
    assert study.best_trial.params is not None
