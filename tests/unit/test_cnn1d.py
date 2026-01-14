"""
CNN1D unit tests.
"""

from collections.abc import Callable

import pytest
import torch
from pydantic import ValidationError

from preference_dynamics.models import CNN1DConfig, CNN1DPredictor, num_parameters


@pytest.mark.parametrize(
    "kwargs,expected_exception",
    [
        (
            {"filters": [32, 64], "kernel_sizes": [5], "features": [16]},
            ValueError,
        ),
        (
            {"filters": [], "kernel_sizes": [5], "features": [16]},
            ValidationError,
        ),
        (
            {"filters": [32], "kernel_sizes": [], "features": [16]},
            ValidationError,
        ),
        (
            {"filters": [32], "kernel_sizes": [5], "features": []},
            ValidationError,
        ),
        (
            {"filters": [32, 0], "kernel_sizes": [5, 3], "features": [16]},
            ValueError,
        ),
        (
            {"filters": [32], "kernel_sizes": [0], "features": [16]},
            ValueError,
        ),
        (
            {"filters": [32], "kernel_sizes": [5], "features": [16], "dropout": 1.5},
            ValidationError,
        ),
        (
            {"filters": [32], "kernel_sizes": [5], "features": [16], "dropout": -0.1},
            ValidationError,
        ),
        (
            {"filters": [32], "kernel_sizes": [5], "features": [16], "in_channels": 0},
            ValidationError,
        ),
    ],
)
def test_config_invalid_cases(
    kwargs: dict[str, object], expected_exception: type[Exception]
) -> None:
    base = {"model_name": "test", "in_channels": 4, "dropout": 0.2}
    with pytest.raises(expected_exception):
        CNN1DConfig(**base | kwargs)


def test_config_creates_layers(
    cnn1d_batch_input: Callable[..., tuple[CNN1DConfig, torch.Tensor]],
) -> None:
    config, _ = cnn1d_batch_input()
    model = CNN1DPredictor(config=config)
    assert model.config == config
    assert len(model.conv_blocks) == len(config.filters)
    assert len(model.fc_layers) == len(config.features)


@pytest.mark.parametrize(
    ("batch_size", "seq_len"),
    [(1, 48), (2, 64), (4, 96)],
)
def test_forward_shapes(
    cnn1d_batch_input: Callable[..., tuple[CNN1DConfig, torch.Tensor]],
    batch_size: int,
    seq_len: int,
) -> None:
    config, inputs = cnn1d_batch_input(batch_size=batch_size, seq_len=seq_len, in_channels=4)
    model = CNN1DPredictor(config=config)
    outputs = model.forward(inputs)
    assert outputs.shape == (batch_size, config.features[-1])


def test_forward_is_differentiable(
    cnn1d_batch_input: Callable[..., tuple[CNN1DConfig, torch.Tensor]],
) -> None:
    config, inputs = cnn1d_batch_input(batch_size=2, seq_len=80, in_channels=4)
    inputs.requires_grad_(True)
    model = CNN1DPredictor(config=config)
    loss = model.forward(inputs).sum()
    loss.backward()
    assert inputs.grad is not None


def test_num_parameters_reported_after_lazy_init(
    cnn1d_batch_input: Callable[..., tuple[CNN1DConfig, torch.Tensor]],
) -> None:
    config, inputs = cnn1d_batch_input(batch_size=2, seq_len=64, in_channels=4)
    model = CNN1DPredictor(config=config)
    model.eval()
    _ = model.forward(inputs)
    assert isinstance(num_parameters(model), int)
    assert num_parameters(model) > 0
