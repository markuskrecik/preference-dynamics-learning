"""
Sagemaker training script for CNN1DResidualPredictor.
"""

import argparse
from pathlib import Path

from preference_dynamics.data.adapters import (
    ParameterICForecastTargetAdapter,
    StateFeatureInputAdapter,
)
from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.models import CNN1DResidualConfig, CNN1DResidualPredictor, save_model
from preference_dynamics.schemas import TrainerConfig
from preference_dynamics.training import Trainer
from preference_dynamics.utils import to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--filters", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 3])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_model(
    in_channels: int,
    out_dim: int,
    args: argparse.Namespace,
) -> CNN1DResidualPredictor:
    config = CNN1DResidualConfig(
        model_name="cnn1d_residual_sagemaker",
        in_channels=in_channels,
        filters=args.filters,
        kernel_sizes=args.kernel_sizes,
        hidden_dims=args.hidden_dims,
        out_dim=out_dim,
        dropout=args.dropout,
    )
    return CNN1DResidualPredictor(config=config)


def train(args: argparse.Namespace) -> None:
    data_config = DataConfig(
        data_dir=args.data_dir,
        load_if_exists=True,
        input_adapter=StateFeatureInputAdapter(),
        target_adapter=ParameterICForecastTargetAdapter(),
        batch_size=args.batch_size,
        shuffle_train=True,
        seed=args.seed,
    )
    dm = DataManager(config=data_config).setup()
    sample = dm.splits["train"][0]
    inputs = data_config.input_adapter.get_inputs(sample)
    in_channels = data_config.input_adapter.n_inputs(sample)
    out_dim = data_config.target_adapter.n_targets(sample)

    model = _build_model(in_channels, out_dim, args)

    trainer_config = TrainerConfig(
        loss_function="mse",
        learning_rate=args.lr,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_best=True,
    )
    trainer = Trainer(model=model, config=trainer_config)
    trainer.fit(dm.train_dataloader, dm.val_dataloader)
    trainer.load_checkpoint("best")

    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    inputs = to_device(inputs, device)
    save_model(model, model_path / "model.pt", trace=True, example_kwarg_inputs=inputs)
    (model_path / "model_config.json").write_text(model.config.model_dump_json())


if __name__ == "__main__":
    args = parse_args()
    train(args)
