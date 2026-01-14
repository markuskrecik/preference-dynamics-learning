"""
Trainer class for neural network parameter prediction models.
"""

import logging
import random
import time
from pathlib import Path
from typing import Literal, Self

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from preference_dynamics.models import MODEL_REGISTRY, PredictorModel, num_parameters
from preference_dynamics.schemas import (
    TrainerConfig,
    TrainingHistory,
)
from preference_dynamics.utils import (
    assemble_checkpoint_path,
    if_logging,
    parse_checkpoint_path,
    stack_dict_tensors,
    to_device,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    """
    Unified training class for neural network parameter prediction models.

    Handles training loop, validation, checkpointing, and progress tracking.
    Works with any PyTorch model that inherits from the PredictorModel ABC.

    Attributes:
        model: PyTorch model inheriting PredictorModel
        config: Training configuration
        device: Training device (cpu/cuda/mps)
        optimizer: PyTorch optimizer
        criterion: Loss function
        best_loss: Best loss seen so far
        patience_counter: Counter for early stopping
        checkpoint_dir: Directory for saving checkpoints
        current_epoch: Current epoch number (for checkpointing)

    Example:
        ```python
        from preference_dynamics.training import Trainer, TrainerConfig
        from preference_dynamics.models import CNN1DPredictor, CNN1DConfig

        # Create model
        model_config = CNN1DConfig(
            model_name="cnn1d_n1",
            in_channels=2,
            filters=[128, 256],
            kernel_sizes=[3, 3],
            features=[128, 4],
            dropout=0.3
        )
        model = CNN1DPredictor(config=model_config)

        # Create trainer
        trainer_config = TrainerConfig(
            loss_function="mse",
            learning_rate=0.001,
            num_epochs=100,
            early_stopping_patience=10,
            seed=42
        )
        trainer = Trainer(model=model, config=trainer_config)

        # Train model
        history = trainer.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader
        )

        # Evaluate on test set
        test_loss = trainer.test(test_loader)
        ```
    """

    def __init__(
        self,
        model: PredictorModel,
        config: TrainerConfig,
    ) -> None:
        self.config = config
        self.run_id = None
        self.current_epoch = 0
        self.patience_counter = 0
        self.max_patience = self.config.early_stopping_patience or float("inf")
        self.best_loss = float("inf")

        self.device = self._init_device(config.device)
        self.model = model
        self.criterion = self._init_loss_function(config.loss_function)

        self._set_random_seed(config.seed)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized Trainer with device={self.device}, checkpoint_dir={self.checkpoint_dir}"
        )

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """
        Initialize Adam optimizer with model parameters.

        Returns:
            Adam optimizer configured with learning_rate and weight_decay from config
        """
        return torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _init_model(self, model: PredictorModel, dataloader: DataLoader) -> PredictorModel:
        """
        Initialize model with a forward pass to trigger lazy initialization.

        Moves model to device and performs a dummy forward pass to initialize
        lazy layers (LazyConv1d, LazyLinear). Logs number of parameters to MLflow.

        Args:
            model: PredictorModel instance

        Returns:
            Initialized model on device

        Raises:
            NotImplementedError: If model type is not supported
        """

        model.to(self.device)
        # with torch.no_grad():
        inputs = dataloader.dataset[0]["inputs"]
        inputs = to_device(inputs, self.device)
        model(**inputs)

        if_logging(mlflow.log_param)("model_num_parameters", num_parameters(model))

        return model

    def _init_device(self, device: Literal["cpu", "cuda", "mps"] | None) -> torch.device:
        """
        Auto-select best available device.

        Args:
            device: Device name or None for auto-detection

        Returns:
            torch.device instance
        """
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _init_loss_function(
        self, loss_function: Literal["mse", "mae", "huber"] | nn.Module
    ) -> nn.Module:
        """
        Get loss function based on config.

        Args:
            loss_function: Loss function name ("mse", "mae", "huber") or nn.Module

        Returns:
            Loss function module

        Raises:
            ValueError: If loss function is unknown
        """
        if isinstance(loss_function, nn.Module):
            return loss_function
        if loss_function == "mse":
            return nn.MSELoss()
        if loss_function == "mae":
            return nn.L1Loss()
        if loss_function == "huber":
            return nn.HuberLoss()
        raise ValueError(
            f"Unknown loss function: {loss_function}. Must be 'mse', 'mae', 'huber', or nn.Module."
        )

    def _set_random_seed(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.

        Sets seeds for torch, numpy, Python random, and cudnn (if available).

        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _clip_grad(self) -> None:
        """
        Clip gradients to prevent explosion.

        Uses gradient_clip_norm from config if set. No-op if gradient_clip_norm is None.
        """
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

    def _check_best_and_patience(self, loss: float) -> tuple[bool, bool]:
        """
        Check if current loss is best and update patience counter.

        Args:
            loss: Current loss value

        Returns:
            Tuple of (is_best, stop) where:
            - is_best: True if current loss is the best so far
            - stop: True if patience counter reached maximum (early stopping triggered)
        """
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        stop = self.patience_counter >= self.max_patience
        return is_best, stop

    def _predict_step(self, batch: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute predictions for one batch.

        Args:
            batch: Batch dictionary with "inputs" key containing dictionary of input tensors

        Returns:
            Predictions tensor of shape (batch_size, output_dim)
        """

        inputs = to_device(batch["inputs"], self.device)
        predictions = self.model(**inputs)
        return predictions

    def _evaluate_step(
        self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor], torch.Tensor | dict[str, torch.Tensor], torch.Tensor
    ]:
        """
        Evaluate one batch (forward pass and loss computation).

        Args:
            batch: Batch dictionary with "inputs" and "targets" keys

        Returns:
            Tuple of (predictions, targets, loss) where predictions/targets may be dicts or tensors
        """

        predictions = self._predict_step(batch)
        targets = to_device(batch["targets"], self.device)

        loss = self.criterion(predictions, targets)

        return predictions, targets, loss

    def _evaluate_epoch(
        self, dataloader: DataLoader
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor], torch.Tensor | dict[str, torch.Tensor], float
    ]:
        """
        Evaluate one epoch for whole dataset.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Tuple of (all_predictions, all_targets, average_loss) where:
            - all_predictions: Tensor, shape (N_samples, output_dim)
            - all_targets: Tensor, shape (N_samples, output_dim)
            - average_loss: scalar float
        """

        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0

        # with torch.no_grad():
        for batch in dataloader:
            predictions, targets, loss = self._evaluate_step(batch)
            all_predictions.append(predictions)
            all_targets.append(targets)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        if isinstance(all_predictions[0], dict):
            out_predictions = stack_dict_tensors(all_predictions)
        else:
            out_predictions = torch.cat(all_predictions, dim=0)

        if isinstance(all_targets[0], dict):
            out_targets = stack_dict_tensors(all_targets)
        else:
            out_targets = torch.cat(all_targets, dim=0)

        return out_predictions, out_targets, avg_loss

    def evaluate(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Evaluate on dataset.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Tuple of (y_pred, y_true, loss) where:
            - y_pred: predictions tensor of shape (N_samples, output_dim)
            - y_true: targets tensor of shape (N_samples, output_dim)
            - loss: average loss (float)
        """
        return self._evaluate_epoch(dataloader)

    def validate(self, val_dataloader: DataLoader) -> float:
        """
        Validate on validation set.

        Args:
            val_dataloader: Validation data loader

        Returns:
            Average validation loss (float)
        """
        _, _, avg_loss = self._evaluate_epoch(val_dataloader)
        return avg_loss

    def test(self, test_dataloader: DataLoader) -> float:
        """
        Test on test set and logs to mlflow.

        Args:
            test_dataloader: Test data loader

        Returns:
            Average test loss (float)
        """
        _, _, avg_loss = self._evaluate_epoch(test_dataloader)
        logger.info(f"Test loss: {avg_loss}")
        return avg_loss

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Generate predictions for whole dataset

        Args:
            dataloader: Data loader for prediction

        Returns:
            Predictions tensor of shape (N_samples, output_dim)
        """

        predictions, _, _ = self._evaluate_epoch(dataloader)
        return predictions

    def _train_step(self, batch: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Perform training step for one batch: forward, backward, optimizer step.

        Args:
            batch: Batch dictionary with keys: "inputs", "targets"

        Returns:
            Loss tensor (scalar) for this batch
        """

        self.optimizer.zero_grad()
        predictions, target, loss = self._evaluate_step(batch)
        loss.backward()

        self._clip_grad()
        self.optimizer.step()

        return loss

    def _train_epoch(self, train_dataloader: DataLoader) -> float:
        """
        Train one epoch on whole dataset.

        Args:
            train_dataloader: Training data loader

        Returns:
            Average training loss (float) for this epoch
        """

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dataloader:
            loss = self._train_step(batch)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ) -> TrainingHistory:
        """
        Run training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader

        Returns:
            TrainingHistory object with training results
        """
        self.run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        train_losses = []
        val_losses = []
        epoch_times = []
        best_epoch = self.current_epoch
        self.model = self._init_model(self.model, train_dataloader)
        self.optimizer = self._init_optimizer()

        logger.info(
            f"Starting training for {self.config.num_epochs} epochs with run_id={self.run_id}"
        )
        pbar = tqdm(
            range(self.current_epoch, self.config.num_epochs),
            desc="Training",
            miniters=1,
            smoothing=1.0,
        )
        for epoch in pbar:
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss = self._train_epoch(train_dataloader)
            train_losses.append(train_loss)
            if_logging(mlflow.log_metric)("train_loss", train_loss, step=epoch)

            val_loss: float | None = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
                if_logging(mlflow.log_metric)("val_loss", val_loss, step=epoch)

            loss = val_loss or train_loss

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            if_logging(mlflow.log_metric)("epoch_time", epoch_time, step=epoch)

            is_best, stop = self._check_best_and_patience(loss)

            if is_best:
                best_epoch = epoch
                if self.config.save_best:
                    self.save_checkpoint("best")

            pbar.set_postfix(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time": epoch_time,
                }
            )
            if stop:
                logger.info(f"Early stopping at epoch {epoch} (patience: {self.patience_counter})")
                break

        self.save_checkpoint("last")

        if_logging(mlflow.log_metrics)(
            {
                "best_epoch": best_epoch,
                "best_loss": self.best_loss,
                "total_epochs": len(train_losses),
            }
        )

        history = TrainingHistory(
            train_loss=train_losses,
            val_loss=val_losses,
            epoch_times=epoch_times,
            best_epoch=best_epoch,
            best_loss=self.best_loss,
            total_epochs=len(train_losses),
            converged=stop,
        )
        return history

    @if_logging
    def save_model_mlflow(self, suffix: str, dataloader: DataLoader) -> None:
        """
        Save MLflow-compatible model as artifact. MLflow models expects a DataFrame as input instead of dict.

        Args:
            suffix: Suffix of model file (without .pt extension)
            dataloader: Data loader to use for input example (takes first sample)
        """
        suffix = str(suffix)
        model_name = f"{self.model.config.model_name}-{suffix}"

        signature = dataloader.dataset.signature

        mlflow.pytorch.log_model(self.model.cpu(), name=model_name, signature=signature)
        self.model.to(self.device)

    @if_logging
    def load_model_mlflow(self, suffix: str) -> PredictorModel:
        """
        Load MLflow model artifact.

        Args:
            suffix: Suffix of model file (without .pt extension)

        Returns:
            Loaded PredictorModel instance (moved to device)
        """
        suffix = str(suffix)
        model_uri = f"runs:/{self.run_id}/{self.model.config.model_name}-{suffix}"
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)
        return self.model

    def _get_checkpoint_path(self, path_stub: str | Path) -> Path:
        """
        Get full path to checkpoint file from path stub.

        Assembles full path using checkpoint_dir, model_name, and run_id from
        trainer state. For valid path stubs, see parse_checkpoint_path().

        Returns:
            Path to checkpoint file (with .pt extension)
        """
        parts = parse_checkpoint_path(path_stub)
        path = assemble_checkpoint_path(
            parts,
            checkpoint_dir=str(self.checkpoint_dir),
            model_name=str(self.model.config.model_name),
            run_id=str(self.run_id),
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def save_checkpoint(self, path_stub: str | Path) -> None:
        """
        Save complete trainer state to checkpoint file.

        Saves model state, optimizer state, training progress, random states,
        and configurations.

        Args:
            path_stub: Path stub (see parse_checkpoint_path() for valid formats)

        Saves:
        - Model state_dict and config
        - Optimizer state_dict
        - Current epoch number
        - Best validation loss
        - Random number generator state (torch, numpy, Python, cuda if available)
        - Trainer configuration
        - Run ID
        """
        checkpoint_path = self._get_checkpoint_path(path_stub)

        random_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if self.device.type == "cuda":
            random_state["torch_cuda"] = torch.cuda.get_rng_state_all()

        checkpoint = {
            "model_config": self.model.config.model_dump(),
            "trainer_config": self.config.model_dump(),
            "run_id": self.run_id,
            "device": self.device.type,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            # "history": self.history.model_dump(),  # TODO: save and load history
            "random_state": random_state,
        }

        torch.save(checkpoint, checkpoint_path)
        if_logging(mlflow.log_param)(f"trainer_checkpoint_{path_stub}", str(checkpoint_path))

    def load_checkpoint(self, path_stub: str | Path) -> Self:
        """
        Load trainer state from checkpoint.

        Restores model, optimizer, training state, and random states from checkpoint.

        Args:
            path_stub: Path stub to load checkpoint from (see parse_checkpoint_path() for valid formats)

        Returns:
            Self

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """

        checkpoint_path = self._get_checkpoint_path(path_stub)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        self.run_id = checkpoint["run_id"]
        self.device = torch.device(checkpoint["device"])

        self.config = TrainerConfig.model_validate(checkpoint["trainer_config"])

        # Restore model
        model_config_dump = checkpoint["model_config"]
        config_cls, model_cls = MODEL_REGISTRY[model_config_dump["model_type"]]

        model_config = config_cls.model_validate(model_config_dump)
        self.model = model_cls(config=model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        # Restore optimizer
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer = self._init_optimizer()
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore training state
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        # self.history = TrainingHistory.model_validate(checkpoint["history"])

        # Restore random state
        if "random_state" in checkpoint:
            random_state = checkpoint["random_state"]
            torch_rng_state = random_state["torch"]
            if isinstance(torch_rng_state, torch.Tensor):
                torch_rng_state = torch_rng_state.cpu()
            torch.set_rng_state(torch_rng_state)
            # TODO: restore cuda rng state
            # if "torch_cuda" in random_state:
            #   torch.cuda.set_rng_state_all(random_state["torch_cuda"])
            np.random.set_state(random_state["numpy"])
            random.setstate(random_state["python"])

        return self

    @classmethod
    def from_checkpoint(cls, path_stub: str | Path) -> Self:
        """
        Create trainer instance from checkpoint.

        Loads checkpoint, creates model and trainer from saved configs, then loads state.
        Note: effectively loads checkpoint twice (once for config, once for state).

        Args:
            path_stub: Path stub to load checkpoint from (see parse_checkpoint_path() for valid formats)

        Returns:
            Trainer instance with loaded state
        """
        parts = parse_checkpoint_path(path_stub)
        checkpoint_path = assemble_checkpoint_path(parts)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model_config_dump = checkpoint["model_config"]
        config_cls, model_cls = MODEL_REGISTRY[model_config_dump["model_type"]]

        model_config = config_cls.model_validate(model_config_dump)
        model = model_cls(config=model_config)

        config = TrainerConfig.model_validate(checkpoint["trainer_config"])

        trainer = cls(model, config)

        # TODO: effectively loads checkpoint twice, can be optimized
        trainer.load_checkpoint(path_stub)

        return trainer
