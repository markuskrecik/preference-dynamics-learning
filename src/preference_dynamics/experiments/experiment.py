"""
Experiment class for orchestrating training runs.
"""

from pathlib import Path
from typing import Self

import mlflow

from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.models import MODEL_REGISTRY, ModelConfig, PredictorModel
from preference_dynamics.schemas import TrainerConfig
from preference_dynamics.training.trainer import Trainer
from preference_dynamics.utils import if_logging


class Experiment:
    """
    Experiment class for orchestrating training runs.

    Encapsulates data management, model, and trainer to provide a unified
    interface for running experiments.

    Attributes:
        data_config: Data configuration
        model_config: Model configuration
        trainer_config: Trainer configuration
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
    ) -> None:
        self.data_config = data_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.trainer = Trainer(model=self._init_model(), config=self.trainer_config)

    def _init_model(self) -> PredictorModel:
        """
        Initialize model from model_config.

        Returns:
            PredictorModel instance

        Raises:
            ValueError: If model type is not supported
        """
        if self.model_config.model_type not in MODEL_REGISTRY:
            raise ValueError(f"Model type {self.model_config.model_type} not supported")
        _, model_cls = MODEL_REGISTRY[self.model_config.model_type]
        return model_cls(config=self.model_config)

    def load_checkpoint(self, checkpoint_path: str | Path) -> Self:
        """
        Load model and trainer state from checkpoint.

        Args:
            checkpoint_path: Path stub to checkpoint from.
                For accepted formats, see `parse_checkpoint_path()`.

        Returns:
            Self
        """
        self.trainer = self.trainer.load_checkpoint(checkpoint_path)
        # self.history = self.trainer.history  # TODO: load history from checkpoint
        return self

    def run(self, checkpoint_path: str | Path | None = None) -> Self:
        """
        Run complete experiment: data setup, model training, and evaluation.

        Args:
            checkpoint_path: Optional checkpoint path stub to resume from.
                For accepted formats, see `parse_checkpoint_path()`.

        Returns:
            Self
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.data_manager = DataManager(config=self.data_config).setup()

        train_dataloader = self.data_manager.train_dataloader
        val_dataloader = self.data_manager.val_dataloader
        test_dataloader = self.data_manager.test_dataloader

        self.history = self.trainer.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        if test_dataloader is None:
            self.loss = (
                self.history.val_loss[-1] if self.history.val_loss else self.history.train_loss[-1]
            )
            return self

        self.trainer.load_checkpoint("best")
        self.loss = self.trainer.test(test_dataloader=test_dataloader)
        self.trainer.save_model_mlflow("best", train_dataloader)
        if_logging(mlflow.log_metric)("test_loss", self.loss)

        return self

    @property
    def model(self) -> PredictorModel:
        return self.trainer.model
