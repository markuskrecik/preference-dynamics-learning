"""
Experiment runner for orchestrating training with different configs,
hyperparameters studies, and MLflow tracking.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import mlflow
from optuna import Study, create_study, create_trial, delete_study, load_study
from optuna.trial import FixedTrial, FrozenTrial, Trial

from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.experiments.experiment import Experiment
from preference_dynamics.models import ModelConfig
from preference_dynamics.schemas import RunnerConfig, TrainerConfig
from preference_dynamics.utils import (
    TrialLike,
    add_prefix,
    ensure_trial_instance,
    if_logging,
    parse_checkpoint_path,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExperimentRunner:
    """
    Base class for experiment runners.
    Orchestrates and tracks complete experiments (data loading, training, evaluation).

    This class handles:
    - Single experiment runs with MLflow logging
    - Hyperparameter studies with Optuna and nested MLflow runs

    Can be used as-is, or subclassed to suggest optuna parameters by implementing the `suggest_parameters` method.

    Attributes:
        runner_config: RunnerConfig instance
        data_config: DataConfig instance
        model_config: ModelConfig instance
        trainer_config: TrainerConfig instance

    Example:
        ```python
        from preference_dynamics.experiments import ExperimentRunner
        from preference_dynamics.schemas import (
            RunnerConfig, DataConfig, TrainerConfig
        )
        from preference_dynamics.models import CNN1DConfig

        # Create configurations
        runner_config = RunnerConfig(
            experiment_name="preference_dynamics_learning",
            mlflow_tracking_uri="sqlite:///mlruns.db"
        )
        data_config = DataConfig(data_dir=Path("data/n1"), batch_size=32, seed=42)
        model_config = CNN1DConfig(model_name="cnn1d_n1", in_channels=2, ...)
        trainer_config = TrainerConfig(loss_function="mse", learning_rate=0.001, ...)

        # Create runner
        runner = ExperimentRunner(
            runner_config=runner_config,
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config
        )

        # Run single experiment
        test_loss = runner.run(run_name="cnn1d_n1_lr_001")

        # Or run hyperparameter study
        class CustomRunner(ExperimentRunner):
            def suggest_parameters(self, trial):
                self.trainer_config.learning_rate = trial.suggest_float("lr", 0.001, 0.01)

        custom_runner = CustomRunner(...)
        study = custom_runner.run_study("lr_sweep", n_trials=10)
        ```
    """

    def __init__(
        self,
        runner_config: RunnerConfig,
        data_config: DataConfig,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
    ) -> None:
        self.config = runner_config
        self.data_config = data_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.study: Study | None = None
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def suggest_parameters(self, trial: Trial) -> None:
        """
        Suggest parameters for optuna study.

        Override this method in subclasses to customize hyperparameter search.

        Args:
            trial: Optuna trial object

        Example:
        ```python
        class Runner(ExperimentRunner):
            def suggest_parameters(self, trial):
                self.trainer_config.learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01)

        runner = Runner(runner_config, data_config, model_config, trainer_config)
        runner.run_study("study_name", n_trials=10)
        ```
        """
        logger.warning("Method `suggest_parameters` not implemented. No parameter scans performed.")

    def return_values(self, experiment: Experiment) -> Sequence[float | int]:
        """
        Return optimization values for optuna study trials.

        Override this method to customize what values are optimized. Use in conjunction
        with self.run_study(directions=Sequence[Literal["minimize", "maximize"]]).
        By default, returns (test) loss.

        Args:
            experiment: Experiment instance of run

        Returns:
            Sequence of values to optimize over

        Example:
        ```python
        class Runner(ExperimentRunner):
            def return_values(self, experiment):
                return (experiment.loss,)
        ```
        """
        return (experiment.loss,)

    def run(self, run_name: str, trial_params: dict[str, Any] | None = None) -> Experiment:
        """
        Run single experiment with MLflow logging.

        Args:
            run_name: MLflow run name (must be unique within experiment)
            trial_params: Inject optional trial parameters to override deeply nested param suggestions

        Returns:
            Experiment instance

        Raises:
            RuntimeError: If MLflow tracking fails (but experiment continues)
        """
        trial = ensure_trial_instance(trial_params)  # TODO: may be obsolete

        with mlflow.start_run(run_name=run_name):
            self._log_parameters(trial)
            mlflow.set_tags(self.config.tags)

            experiment = Experiment(
                data_config=self.data_config,
                model_config=self.model_config,
                trainer_config=self.trainer_config,
            )
            experiment.run()
        return experiment

    def load_checkpoint(self, checkpoint_path: str | Path) -> Experiment:
        """
        Load experiment from checkpoint.

        Args:
            checkpoint_path: Path stub to checkpoint from.
                For accepted formats, see `parse_checkpoint_path()`.

        Returns:
            Experiment instance with loaded model and trainer state
        """
        experiment = Experiment(
            data_config=self.data_config,
            model_config=self.model_config,
            trainer_config=self.trainer_config,
        )
        experiment.load_checkpoint(checkpoint_path)
        return experiment

    def run_from_checkpoint(self, checkpoint_path: str | Path) -> Experiment:
        """
        Loads checkpoint, extracts run_id, and continues training in the same
        MLflow run.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Experiment instance after continuing training
        """
        experiment = self.load_checkpoint(checkpoint_path)

        parts = parse_checkpoint_path(checkpoint_path)
        run_id = parts[-2]
        with mlflow.start_run(run_id=run_id):
            experiment.run()
        return experiment

    def _objective(self, trial: Trial) -> Sequence[float | int]:
        """
        Objective function for Optuna optimization (wrapped in nested MLflow run).

        Creates nested MLflow run, suggests parameters, runs experiment, and returns
        optimization values.

        Args:
            trial: Optuna trial object

        Returns:
            Sequence of values to optimize (from return_values method)
        """
        self.suggest_parameters(trial)

        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        run_name = f"Trial {trial.number}: {params_str}"
        logger.info(f"Run: {run_name}")

        with mlflow.start_run(nested=True, run_name=run_name) as child_run:
            self._log_parameters(trial)
            mlflow.set_tags(self.config.tags)
            mlflow.set_tag("mlflow.parentRunId", trial.study.user_attrs.get("parent_run_id", ""))
            trial.set_user_attr("mlflow_run_id", child_run.info.run_id)
            trial.set_user_attr("mlflow_run_name", run_name)

            experiment = Experiment(
                data_config=self.data_config,
                model_config=self.model_config,
                trainer_config=self.trainer_config,
            )
            experiment.run()
        return self.return_values(experiment)

    def run_study(
        self,
        study_name: str,
        n_trials: int,
        n_jobs: int = 1,
        directions: Sequence[Literal["minimize", "maximize"]] = ("minimize",),
    ) -> Study:
        """
        Run hyperparameter study with Optuna and nested MLflow runs.

        Args:
            study_name: Name for Optuna study and MLflow parent run
            n_trials: Number of hyperparameter trials to run
            n_jobs: Number of parallel jobs (default: 1, or -1 for all available CPUs)
            storage: Optuna storage backend (default: "sqlite:///optuna.db")

        Raises:
            RuntimeError: If Optuna study creation fails
            ValueError: If n_trials < 1, n_jobs < -1, or n_jobs == 0
        """
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError("n_jobs must be >= 1, or -1 for all available CPUs")

        with mlflow.start_run(run_name=study_name) as parent_run:
            mlflow.set_tags(self.config.tags)

            self.study = create_study(
                study_name=study_name,
                storage=self.config.optuna_storage,
                load_if_exists=True,
                directions=directions,
            )
            self.study.set_user_attr("parent_run_id", parent_run.info.run_id)
            self.study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs)

            best_trial = self.study.best_trial
            best_run_name = best_trial.user_attrs.get("mlflow_run_name", "")
            mlflow.log_metric("best_trial_loss", best_trial.value)
            mlflow.log_params(add_prefix(best_trial.params, "best_trial"))
            mlflow.log_params(
                add_prefix(
                    {
                        "run_id": best_trial.user_attrs.get("mlflow_run_id", ""),
                        "run_name": best_run_name,
                    },
                    "best_trial",
                ),
            )
            logging.info(f"Best trial: {best_run_name}, loss: {best_trial.value}")

        return self.study

    def load_study(self, study_name: str) -> Study:
        """
        Load Optuna study from storage by study_name.

        Args:
            study_name: Name of the study to load

        Returns:
            Optuna Study instance
        """
        self.study = load_study(study_name=study_name, storage=self.config.optuna_storage)
        return self.study

    def delete_study(self, study_name: str) -> None:
        """
        Delete Optuna study from storage by study_name.

        Args:
            study_name: Name of the study to delete
        """
        delete_study(study_name=study_name, storage=self.config.optuna_storage)

    def delete_trials(
        self, study_name: str, trial_ids: Sequence[int], replace: bool = False
    ) -> Study:
        """
        Pseudo-delete trials by creating a new study without specified trials.

        Since Optuna does not support trial deletion, this creates a new study
        containing only the trials not in trial_ids. If replace=False, new studies
        are numbered sequentially as study_name_1, study_name_2, etc.

        Args:
            study_name: Name of the study to delete trials from
            trial_ids: List of trial IDs (Trial.number) to exclude from new study
            replace: Whether to replace the old study. May cause data loss if True. You have been warned.

        Returns:
            New Study instance without deleted trials
        """
        study = load_study(study_name=study_name, storage=self.config.optuna_storage)
        old_trials = study.get_trials(deepcopy=False)
        new_trials = []
        for t in old_trials:
            if t.number not in trial_ids:
                new_trial = create_trial(
                    values=t.values,
                    params=t.params,
                    distributions=t.distributions,
                    user_attrs=t.user_attrs,
                    system_attrs=t.system_attrs,
                    intermediate_values=t.intermediate_values,
                )
                new_trial.datetime_start = t.datetime_start
                new_trial.datetime_complete = t.datetime_complete
                new_trials.append(new_trial)

        if replace:
            new_study_name = study_name
            delete_study(study_name=study_name, storage=study._storage)
        else:
            name_parts = study_name.split("_")
            try:
                study_num = int(name_parts[-1]) + 1
                new_study_name = "_".join(name_parts[:-1] + [str(study_num)])
            except ValueError:
                new_study_name = f"{study_name}_1"

        new_study = create_study(
            storage=study._storage,
            sampler=study.sampler,
            pruner=study.pruner,
            study_name=new_study_name,
            load_if_exists=False,
            directions=study.directions,
        )
        new_study.add_trials(new_trials)
        self.study = new_study
        return self.study

    @if_logging
    def _log_parameters(self, trial: TrialLike) -> None:
        """
        Logs model config, trainer config, and trial parameters (if applicable)
        with corresponding prefixes.

        Args:
            trial: TrialLike object (Trial, FrozenTrial, FixedTrial, dict, or None)
        """

        params: dict[str, Any] = {}

        model_params = self.model_config.model_dump()
        trainer_params = self.trainer_config.model_dump()
        params.update(add_prefix(model_params, "model"))
        params.update(add_prefix(trainer_params, "trainer"))

        if isinstance(trial, (Trial, FrozenTrial, FixedTrial)):
            params.update(add_prefix(trial.params, "trial"))
        mlflow.log_params(params)
