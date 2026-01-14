import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from preference_dynamics.data.adapters import StateFeatureInputAdapter
from preference_dynamics.sagemaker import inference_cnn1d_residual as inference
from preference_dynamics.sagemaker import train_cnn1d_residual as train
from preference_dynamics.schemas import TimeSeriesSample


@pytest.mark.slow
def test_train_script(
    data_dir: Path,
    tmp_path: Path,
    sample_data: tuple[list[TimeSeriesSample], list[TimeSeriesSample], list[TimeSeriesSample]],
) -> None:
    model_dir = tmp_path / "model"
    checkpoint_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    args = argparse.Namespace(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        checkpoint_dir=str(checkpoint_dir),
        epochs=2,
        patience=1,
        lr=1e-2,
        batch_size=2,
        filters=[1],
        kernel_sizes=[1],
        hidden_dims=[1],
        dropout=0.1,
        seed=42,
    )

    train.train(args)

    assert (model_dir / "model.pt").exists()
    assert (model_dir / "model_config.json").exists()

    model_config = json.loads((model_dir / "model_config.json").read_text())
    assert model_config["model_name"] == "cnn1d_residual_sagemaker"
    assert model_config["in_channels"] > 0
    assert model_config["out_dim"] > 0


def test_inference_model_fn(model_dir: Path) -> None:
    model, config = inference.model_fn(str(model_dir))

    assert model is not None
    assert isinstance(config, dict)
    assert "in_channels" in config
    assert "out_dim" in config
    assert model.training is False


def test_inference_predict_fn(model_dir: Path) -> None:
    model, config = inference.model_fn(str(model_dir))
    in_channels = config["in_channels"]

    input_data = [
        {
            "time_series": np.random.randn(in_channels, 16).tolist(),
            "features": [True, 1.5, 2.5],
        }
    ]

    predictions = inference.predict_fn(input_data, (model, config))

    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert isinstance(predictions[0], list)
    assert len(predictions[0]) == config["out_dim"]

    with pytest.raises(ValueError, match="Trained model got"):
        bad_input = [
            {
                "time_series": np.random.randn(in_channels + 1, 16).tolist(),
                "features": [True, 1.5, 2.5],
            }
        ]
        inference.predict_fn(bad_input, (model, config))


def test_inference_end_to_end(
    model_dir: Path,
    sample_data: tuple[list[TimeSeriesSample], list[TimeSeriesSample], list[TimeSeriesSample]],
) -> None:
    _, _, test_samples = sample_data
    test_sample = test_samples[0]

    model, config = inference.model_fn(str(model_dir))

    adapter = StateFeatureInputAdapter()
    features = adapter.get_inputs(test_sample)["x_feat"].cpu().numpy().tolist()

    sample_dict = {
        "time_series": test_sample.time_series.tolist(),
        "features": features,
    }

    request_body = json.dumps([sample_dict])
    input_data = inference.input_fn(request_body, "application/json")
    predictions = inference.predict_fn(input_data, (model, config))
    response, content_type = inference.output_fn(predictions, "application/json")

    assert isinstance(response, str)
    assert content_type == "application/json"
    parsed = json.loads(response)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert isinstance(parsed[0], list)
    assert len(parsed[0]) == config["out_dim"]


def test_inference_batch_processing(model_dir: Path) -> None:
    model, config = inference.model_fn(str(model_dir))
    in_channels = config["in_channels"]

    samples = [
        {
            "time_series": np.random.randn(in_channels, 16).tolist(),
            "features": [True, 1.5, 2.5],
        }
        for _ in range(3)
    ]

    request_body = json.dumps(samples)
    input_data = inference.input_fn(request_body, "application/json")
    predictions = inference.predict_fn(input_data, (model, config))
    response, _ = inference.output_fn(predictions, "application/json")

    parsed = json.loads(response)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    assert all(isinstance(p, list) and len(p) == config["out_dim"] for p in parsed)
