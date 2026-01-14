"""
Unit tests for Trainer class.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from preference_dynamics.models.cnn1d import CNN1DPredictor
from preference_dynamics.schemas import TrainerConfig, TrainingHistory
from preference_dynamics.training.trainer import Trainer


def test_trainer_config_validates() -> None:
    config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.01,
        num_epochs=3,
        early_stopping_patience=2,
        gradient_clip_norm=1.0,
        device="cpu",
        seed=0,
        checkpoint_dir="./checkpoints",
        save_best=True,
    )
    assert config.loss_function == "mse"
    assert config.gradient_clip_norm == 1.0


def test_trainer_config_invalid_learning_rate() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(loss_function="mse", learning_rate=0.0, num_epochs=1)


@pytest.mark.skip(reason="TODO: Fix test")
def test_gradient_clipping_applied(
    trainer_checkpoint_env: tuple[CNN1DPredictor, DataLoader, DataLoader, Path],
) -> None:
    cnn1d_model, train_loader, _val_loader, checkpoint_dir = trainer_checkpoint_env
    config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.01,
        num_epochs=1,
        gradient_clip_norm=0.5,
        checkpoint_dir=str(checkpoint_dir),
    )
    trainer = Trainer(model=cnn1d_model, config=config)

    batch = next(iter(train_loader))
    trainer._train_step(batch)

    total_grad_norm = torch.sqrt(
        sum(p.grad.detach().norm() ** 2 for p in trainer.model.parameters() if p.grad is not None)
    )
    np.testing.assert_array_less(total_grad_norm.cpu().numpy(), config.gradient_clip_norm + 1e-6)


def test_checkpoint_roundtrip(
    trainer_checkpoint_env: tuple[CNN1DPredictor, DataLoader, DataLoader, Path],
) -> None:
    cnn1d_model, train_loader, val_loader, checkpoint_dir = trainer_checkpoint_env
    config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.01,
        num_epochs=2,
        checkpoint_dir=str(checkpoint_dir),
        save_best=True,
    )
    trainer = Trainer(model=cnn1d_model, config=config)
    history = trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)
    checkpoint_stub = "test_checkpoint"
    trainer.save_checkpoint(checkpoint_stub)

    restored = Trainer(model=cnn1d_model, config=config)
    restored.load_checkpoint(checkpoint_stub)

    assert restored.best_loss == trainer.best_loss
    assert history.train_loss


def test_loss_decreases_on_toy_data(
    trainer_checkpoint_env: tuple[CNN1DPredictor, DataLoader, DataLoader, Path],
) -> None:
    cnn1d_model, train_loader, val_loader, checkpoint_dir = trainer_checkpoint_env
    config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.02,
        num_epochs=4,
        early_stopping_patience=3,
        checkpoint_dir=str(checkpoint_dir),
    )
    trainer = Trainer(model=cnn1d_model, config=config)

    history = trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)

    assert history.train_loss[0] >= history.train_loss[-1]
    assert isinstance(history, TrainingHistory)


def test_trainer_early_stopping_and_checkpoint_best(
    trainer_checkpoint_env: tuple[CNN1DPredictor, DataLoader, DataLoader, Path],
) -> None:
    cnn1d_model, train_loader, val_loader, checkpoint_dir = trainer_checkpoint_env
    config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.01,
        num_epochs=3,
        early_stopping_patience=1,
        checkpoint_dir=str(checkpoint_dir),
        save_best=True,
    )
    trainer = Trainer(model=cnn1d_model, config=config)

    def _constant_validate(_loader: DataLoader) -> float:
        return 1.0

    trainer.validate = _constant_validate
    history = trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)

    best_path = (
        Path(checkpoint_dir) / str(cnn1d_model.config.model_name) / str(trainer.run_id) / "best.pt"
    )
    last_path = (
        Path(checkpoint_dir) / str(cnn1d_model.config.model_name) / str(trainer.run_id) / "last.pt"
    )
    assert best_path.exists()
    assert last_path.exists()
    assert history.converged is True


def test_checkpoint_path_errors(
    trainer_checkpoint_env: tuple[CNN1DPredictor, DataLoader, DataLoader, Path],
) -> None:
    cnn1d_model, _train_loader, _val_loader, checkpoint_dir = trainer_checkpoint_env
    config = TrainerConfig(
        loss_function="mse",
        learning_rate=0.01,
        num_epochs=1,
        checkpoint_dir=str(checkpoint_dir),
    )
    trainer = Trainer(model=cnn1d_model, config=config)

    with pytest.raises(ValueError):
        trainer._get_checkpoint_path(Path("a/b/c/d/e"))
