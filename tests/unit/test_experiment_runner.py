"""
Unit tests for ExperimentRunner class.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.experiments.runner import ExperimentRunner
from preference_dynamics.models import CNN1DConfig
from preference_dynamics.schemas import RunnerConfig, TrainerConfig


def test_runner_config_validation_defaults() -> None:
    config = RunnerConfig(experiment_name="test")
    assert config.optuna_storage == "sqlite:///optuna.db"
    assert config.log_interval == 1


def test_runner_config_invalid_name() -> None:
    with pytest.raises(ValidationError):
        RunnerConfig(experiment_name="")


def test_run_raises_on_invalid_trial_params(
    cnn1d_config: CNN1DConfig, simple_trainer_config: TrainerConfig, tmp_path: Path
) -> None:
    runner = ExperimentRunner(
        runner_config=RunnerConfig(
            experiment_name="test", mlflow_tracking_uri="sqlite:///:memory:"
        ),
        data_config=DataConfig(data_dir=str(tmp_path / "data"), load_if_exists=False),
        model_config=cnn1d_config,
        trainer_config=simple_trainer_config,
    )
    with pytest.raises(TypeError):
        runner.run(run_name="bad_trial", trial_params=["not", "valid"])


# TODO: check test
def test_run_uses_mlflow_context_and_returns_experiment(
    cnn1d_config: CNN1DConfig,
    simple_trainer_config: TrainerConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class DummyRun:
        def __enter__(self) -> "DummyRun":
            calls.append("start_run")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            calls.append("end_run")

    class DummyExperiment:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.loss = None

        def run(self) -> "DummyExperiment":
            self.loss = 0.0
            return self

    monkeypatch.setattr(
        "preference_dynamics.experiments.runner.mlflow.start_run", lambda **_: DummyRun()
    )
    monkeypatch.setattr(
        "preference_dynamics.experiments.runner.mlflow.set_tags", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("preference_dynamics.experiments.runner.Experiment", DummyExperiment)

    runner = ExperimentRunner(
        runner_config=RunnerConfig(
            experiment_name="test", mlflow_tracking_uri="sqlite:///:memory:"
        ),
        data_config=DataConfig(data_dir=str(tmp_path / "data"), load_if_exists=False),
        model_config=cnn1d_config,
        trainer_config=simple_trainer_config,
    )

    experiment = runner.run(run_name="ok")

    assert experiment.loss == 0.0
    assert calls == ["start_run", "end_run"]


def test_run_study_validates_jobs_and_trials(
    cnn1d_config: CNN1DConfig, simple_trainer_config: TrainerConfig, tmp_path: Path
) -> None:
    runner = ExperimentRunner(
        runner_config=RunnerConfig(
            experiment_name="test", mlflow_tracking_uri="sqlite:///:memory:"
        ),
        data_config=DataConfig(data_dir=str(tmp_path / "data"), load_if_exists=False),
        model_config=cnn1d_config,
        trainer_config=simple_trainer_config,
    )
    with pytest.raises(ValueError):
        runner.run_study(study_name="s", n_trials=0)
    with pytest.raises(ValueError):
        runner.run_study(study_name="s", n_trials=1, n_jobs=0)
