"""
Fixtures for testing.
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.models.cnn1d import CNN1DPredictor
from preference_dynamics.models.schemas import CNN1DConfig, ModelConfig
from preference_dynamics.schemas import (
    ICVector,
    ODEConfig,
    ODESolverConfig,
    ParameterVector,
    SolverConfig,
    TimeSeriesSample,
    TrainerConfig,
)
from preference_dynamics.utils import get_diagonal_indices

# ============================================================================
# Solver fixtures
# ============================================================================


@pytest.fixture
def random_sample_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def random_parameters_n3() -> ParameterVector:
    params = np.random.randn(24)
    for idx in get_diagonal_indices(3):
        params[idx] = np.abs(params[idx])
    return ParameterVector(n_actions=3, values=params)


@pytest.fixture
def random_initial_conditions_n3() -> ICVector:
    return ICVector(n_actions=3, values=np.random.randn(6))


@pytest.fixture
def random_ode_config_n3(
    random_parameters_n3: ParameterVector, random_initial_conditions_n3: ICVector
) -> ODEConfig:
    return ODEConfig(
        parameters=random_parameters_n3,
        initial_conditions=random_initial_conditions_n3,
    )


@pytest.fixture
def random_solver_config() -> SolverConfig:
    time_span = np.sort(np.random.uniform(0.0, 100.0, size=2))
    n_time_points = np.random.randint(100, 1000)
    solver_method = np.random.choice(["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"])
    rtol = np.random.uniform(1e-6, 1e-3)
    atol = np.random.uniform(1e-9, 1e-6)
    events = None
    return SolverConfig(
        time_span=time_span,
        n_time_points=n_time_points,
        solver_method=solver_method,
        rtol=rtol,
        atol=atol,
        events=events,
    )


@pytest.fixture
def random_config_n3(
    random_ode_config_n3: ODEConfig, random_solver_config: SolverConfig
) -> ODESolverConfig:
    return ODESolverConfig(ode=random_ode_config_n3, solver=random_solver_config)


@pytest.fixture
def random_time_series_n3(
    random_sample_id: str,
    random_config_n3: ODESolverConfig,
) -> TimeSeriesSample:
    time_series = np.random.randn(
        2 * random_config_n3.ode.n_actions, random_config_n3.solver.n_time_points
    )
    time_series[3:6, :] = np.abs(time_series[3:6, :])
    time_points = np.linspace(
        random_config_n3.solver.time_span[0],
        random_config_n3.solver.time_span[1],
        random_config_n3.solver.n_time_points,
    )
    return TimeSeriesSample(
        sample_id=random_sample_id,
        time_series=time_series,
        time_points=time_points,
        config=random_config_n3,
        metadata={"test_key": "test_val"},
    )


@pytest.fixture
def make_time_series_sample() -> Callable[..., TimeSeriesSample]:
    def _make(
        time_series: np.ndarray,
        *,
        time_points: np.ndarray | None = None,
        n_actions: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimeSeriesSample:
        channels, length = time_series.shape
        actions = n_actions or channels // 2
        params = ParameterVector(values=np.ones(2 * actions + 2 * actions**2))
        ic = ICVector(values=np.zeros(2 * actions))
        solver_config = SolverConfig(time_span=(0.0, float(length - 1)), n_time_points=length)
        ode_config = ODEConfig(parameters=params, initial_conditions=ic, random_seed=0)
        config = ODESolverConfig(ode=ode_config, solver=solver_config)
        points = time_points if time_points is not None else np.linspace(0.0, length - 1, length)
        return TimeSeriesSample(
            time_series=time_series,
            time_points=points,
            config=config,
            metadata=metadata or {},
        )

    return _make


# ----------------------------------------------------------------------------
# Transformer fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def steady_state_samples() -> dict[str, np.ndarray]:
    steady = np.ones((2, 40))
    near_steady = np.ones((2, 60))
    near_steady[:, -10:] += 0.01
    drifting = np.vstack(
        [
            np.linspace(0.0, 1.0, 50),
            np.concatenate([np.zeros(25), np.linspace(0.0, 1.0, 25)]),
        ]
    )
    short = np.ones((2, 3))
    return {
        "steady": steady,
        "near_steady": near_steady,
        "drifting": drifting,
        "short": short,
    }


@pytest.fixture
def peaks_sample(make_time_series_sample: Callable[..., TimeSeriesSample]) -> TimeSeriesSample:
    time_points = np.linspace(0.0, 29.0, 30)
    time_series = np.zeros((2, 30))
    time_series[0, [5, 12, 19, 26]] = [2.5, 2.2, 2.4, 2.3]
    time_series[1, [4, 11, 18, 25]] = [1.8, 1.7, 1.9, 1.85]
    return make_time_series_sample(time_series, time_points=time_points)


# TODO: check if fixture used
@pytest.fixture
def data_dir_with_raw_and_processed(
    tmp_path: Path, random_time_series_n3: TimeSeriesSample
) -> Path:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    samples = [random_time_series_n3.model_copy() for _ in range(6)]
    from preference_dynamics.data.schemas import DataConfig  # noqa: PLC0415

    io_handler = DataConfig(data_dir=str(data_dir)).io_handler
    io_handler.save(samples[:4], raw_dir / "train")
    io_handler.save(samples[:3], processed_dir / "train")
    io_handler.save(samples[3:5], processed_dir / "val")
    io_handler.save(samples[5:], processed_dir / "test")
    return data_dir


# ============================================================================
# Model fixtures
# ============================================================================


@pytest.fixture
def simple_model_config() -> ModelConfig:
    return ModelConfig(
        model_type="cnn1d",
        model_name="test_model",
    )


class SimpleModel(PredictorModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.fc = torch.nn.LazyLinear(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @property
    def config(self) -> ModelConfig:
        return self._config


@pytest.fixture
def simple_model(simple_model_config: ModelConfig) -> PredictorModel:
    model = SimpleModel(config=simple_model_config)
    yield model
    del model


@pytest.fixture
def mock_model_config() -> ModelConfig:
    return ModelConfig(
        model_type="mock",
        model_name="test_mock_model",
    )


@pytest.fixture
def mock_model(mock_model_config: ModelConfig) -> PredictorModel:
    model = MagicMock(spec=PredictorModel)
    model.forward = MagicMock(return_value=torch.randn(2, 4))
    model.config = mock_model_config
    return model


@pytest.fixture
def cnn1d_config() -> CNN1DConfig:
    return CNN1DConfig(
        model_name="test_cnn1d_model",
        in_channels=2,
        filters=[4],
        kernel_sizes=[3],
        features=[4],
        dropout=0.2,
    )


@pytest.fixture
def cnn1d_model(cnn1d_config: CNN1DConfig) -> CNN1DPredictor:
    model = CNN1DPredictor(config=cnn1d_config)
    yield model
    del model


@pytest.fixture
def cnn1d_batch_input() -> Callable[..., tuple[CNN1DConfig, torch.Tensor]]:
    def _make(
        batch_size: int = 2, seq_len: int = 64, in_channels: int = 4
    ) -> tuple[CNN1DConfig, torch.Tensor]:
        config = CNN1DConfig(
            model_name="cnn1d_test",
            in_channels=in_channels,
            filters=[8, 8],
            kernel_sizes=[5, 3],
            features=[16],
            dropout=0.1,
        )
        inputs = torch.randn(batch_size, in_channels, seq_len)
        return config, inputs

    return _make


# ============================================================================
# Trainer fixtures
# ============================================================================


@pytest.fixture
def simple_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        loss_function="mse",
        learning_rate=0.005,
        num_epochs=2,
        early_stopping_patience=1,
        weight_decay=1e-4,
        gradient_clip_norm=2.0,
        device=None,
        seed=1,
        checkpoint_dir="./checkpoints",
        save_best=True,
    )


@pytest.fixture
def trainer_checkpoint_env(
    tmp_path_factory: pytest.TempPathFactory, cnn1d_model: CNN1DPredictor
) -> tuple[CNN1DPredictor, DataLoader, DataLoader, Path]:
    class _ToyDataset(Dataset):
        def __init__(self, n_samples: int, in_channels: int, seq_len: int, output_dim: int) -> None:
            self.n_samples = n_samples
            self.in_channels = in_channels
            self.seq_len = seq_len
            self.output_dim = output_dim

        def __len__(self) -> int:
            return self.n_samples

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
            torch.manual_seed(idx)
            data = torch.randn(self.in_channels, self.seq_len)
            target = torch.zeros(self.output_dim)
            return {
                "inputs": {"x": data},
                "targets": target,
            }

    checkpoint_dir = tmp_path_factory.mktemp("trainer_checkpoints")
    in_channels = cnn1d_model.config.in_channels
    seq_len = 40
    output_dim = cnn1d_model.config.features[-1]
    train_ds = _ToyDataset(
        n_samples=12, in_channels=in_channels, seq_len=seq_len, output_dim=output_dim
    )
    val_ds = _ToyDataset(
        n_samples=6, in_channels=in_channels, seq_len=seq_len, output_dim=output_dim
    )
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=3)
    return cnn1d_model, train_loader, val_loader, checkpoint_dir
