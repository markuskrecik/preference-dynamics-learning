"""
Sagemaker training script for CNN1DResidualPredictor.
"""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from preference_dynamics.data.adapters import (
    ParameterICForecastTargetAdapter,
    ParameterTargetAdapter,
    StateFeatureInputAdapter,
)
from preference_dynamics.data.dataset import TimeSeriesDataset
from preference_dynamics.data.manager import DataManager
from preference_dynamics.data.schemas import DataConfig
from preference_dynamics.models import CNN1DResidualConfig, CNN1DResidualPredictor
from preference_dynamics.schemas import TimeSeriesSample, TrainerConfig
from preference_dynamics.training import Trainer


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


def _load_samples_from_dir(path: str | Path) -> list[TimeSeriesSample]:
    root = Path(path)
    files = [p for p in sorted(root.glob("*.json")) if p.is_file()]
    samples: list[TimeSeriesSample] = []
    for file in files:
        payload = json.loads(file.read_text())
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError(f"Expected list or object in {file}, got {type(payload)}")
        for item in payload:
            if isinstance(item, str):
                samples.append(TimeSeriesSample.model_validate_json(item))
            else:
                samples.append(TimeSeriesSample.model_validate(item))
    if not samples:
        raise FileNotFoundError(f"No JSON files with samples found in {path}")
    return samples


def _create_dataloader(
    samples: list[TimeSeriesSample],
    data_config: DataConfig,
    split_name: str,
) -> DataLoader:
    dataset = TimeSeriesDataset(
        samples,
        input_adapter=data_config.input_adapter,
        target_adapter=data_config.target_adapter,
    )
    shuffle = data_config.shuffle_train if split_name == "train" else False
    drop_last = split_name == "train"
    loader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return loader


def _build_model(
    target_adapter: ParameterTargetAdapter,
    sample: TimeSeriesSample,
    args: argparse.Namespace,
) -> CNN1DResidualPredictor:
    # input_adapter = StateFeatureInputAdapter()
    # inputs = input_adapter.get_inputs(train_sample)
    in_channels = sample.time_series.shape[0]
    target_dim = target_adapter.n_targets(sample)
    config = CNN1DResidualConfig(
        model_name="cnn1d_residual_sagemaker",
        in_channels=in_channels,
        filters=args.filters,
        kernel_sizes=args.kernel_sizes,
        hidden_dims=args.hidden_dims,
        out_dim=target_dim,
        dropout=args.dropout,
    )
    # metadata = {
    #     "in_channels": in_channels,
    #     # "seq_len": train_sample.time_series.shape[1],
    #     # "n_features": int(inputs["x_feat"].numel()),
    #     # "param_dim": len(train_sample.parameters.values),
    #     # "ic_dim": len(train_sample.initial_conditions.values),
    #     # "forecast_dim": len(target_adapter.get_targets(train_sample))
    #     # - len(train_sample.parameters.values)
    #     # - len(train_sample.initial_conditions.values),
    #     "target_dim": target_dim,
    #     "filters": args.filters,
    #     "kernel_sizes": args.kernel_sizes,
    #     "hidden_dims": args.hidden_dims,
    #     "dropout": args.dropout,
    # }
    return CNN1DResidualPredictor(config=config)  # , metadata


# def _prepare_sagemaker_data(data_dir: str) -> None:
#     """
#     Prepare data from SageMaker channels for DataManager.

#     If multiple channels exist (train/, validation/, test/), copy files to processed/.
#     If single channel exists, copy files from that channel directory to processed/.
#     """
#     data_path = Path(data_dir)
#     processed_dir = data_path / "processed"
#     processed_dir.mkdir(parents=True, exist_ok=True)

#     # Check for multiple channels (train/, validation/, test/)
#     channel_dirs = ["train", "validation", "test"]
#     has_multiple_channels = any((data_path / name).exists() for name in channel_dirs)

#     if has_multiple_channels:
#         # Multiple channels: copy from each channel directory
#         mapping = {"train": "train", "validation": "val", "test": "test"}
#         for channel_name, split_name in mapping.items():
#             channel_dir = data_path / channel_name
#             if channel_dir.exists():
#                 json_files = list(channel_dir.glob("*.json"))
#                 if json_files:
#                     shutil.copy2(json_files[0], processed_dir / f"{split_name}.json")
#     else:
#         # Single channel: find the channel directory and copy files
#         subdirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != "processed"]
#         if subdirs:
#             channel_dir = subdirs[0]
#             for json_file in channel_dir.glob("*.json"):
#                 filename = json_file.name
#                 # Map filenames to split names
#                 if filename.startswith("train"):
#                     target = processed_dir / "train.json"
#                 elif filename.startswith(("val", "validation")):
#                     target = processed_dir / "val.json"
#                 elif filename.startswith("test"):
#                     target = processed_dir / "test.json"
#                 else:
#                     target = processed_dir / filename
#                 shutil.copy2(json_file, target)


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # _prepare_sagemaker_data(args.data_dir)

    data_config = DataConfig(
        data_dir=args.data_dir,
        load_if_exists=True,
        input_adapter=StateFeatureInputAdapter(),
        target_adapter=ParameterICForecastTargetAdapter(),
        batch_size=args.batch_size,
        shuffle_train=True,
        # num_workers=os.cpu_count() or 0,
        # pin_memory=True,
        seed=args.seed,
    )
    dm = DataManager(config=data_config).setup()
    sample = dm.splits["train"][0]

    # data_path = Path(args.data_dir)
    # train_samples = _load_samples_from_dir(data_path / "train")
    # val_samples = _load_samples_from_dir(data_path / "validation")
    # test_samples = _load_samples_from_dir(data_path / "test")

    # train_loader = _create_dataloader(train_samples, data_config, "train")
    # val_loader = _create_dataloader(val_samples, data_config, "validation")
    # test_loader = _create_dataloader(test_samples, data_config, "test")

    model = _build_model(data_config.target_adapter, sample, args)

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
    # trainer.evaluate(val_loader)
    # if test_loader is not None:
    #     trainer.evaluate(test_loader)
    trainer.load_checkpoint("best")
    trainer.model.save(str(model_path / "model.pt"))
    (model_path / "model_config.json").write_text(model.config.model_dump_json())


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
