import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from preference_dynamics.models import CNN1DResidualConfig, CNN1DResidualPredictor


def _load_config(model_dir: Path) -> CNN1DResidualConfig:
    config_path = model_dir / "model_config.json"
    return CNN1DResidualConfig.model_validate_json(config_path.read_text())


def _load_model(model_dir: Path, config: CNN1DResidualConfig) -> CNN1DResidualPredictor:
    model = CNN1DResidualPredictor(config=config)
    state_path = model_dir / "model.pt"
    model.load(state_path)
    return model


def model_fn(model_dir: str) -> CNN1DResidualPredictor:
    model_path = Path(model_dir)
    config = _load_config(model_path)
    model = _load_model(model_path, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def _ensure_list_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return payload
        raise ValueError("JSON array must contain objects")
    raise ValueError("Unsupported JSON payload; expected object or array of objects")


def input_fn(request_body: bytes, content_type: str) -> list[dict[str, Any]]:
    if content_type != "application/json":
        raise ValueError("Only application/json is supported")
    text = request_body.decode("utf-8").strip()
    if not text:
        return []
    payload = json.loads(text)
    return _ensure_list_payload(payload)


def _prepare_tensor(
    sample: dict[str, Any],
    config: CNN1DResidualConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_np = np.asarray(sample["time_series"])
    x_feat_np = np.asarray(sample["features"])

    if x_np.shape[0] != config.in_channels:
        raise ValueError(
            f"Trained model got {x_np.shape[0]} channels, expected {config.in_channels}."
        )

    x = torch.from_numpy(x_np).float().unsqueeze(0)
    x_feat = torch.from_numpy(x_feat_np).float().unsqueeze(0)
    return x, x_feat


# def _split_output(pred: np.ndarray, config: CNN1DResidualConfig) -> dict[str, list[float]]:
#     param_dim = metadata["param_dim"]
#     ic_dim = metadata["ic_dim"]
#     forecast_dim = metadata["forecast_dim"]
#     params = pred[:param_dim].tolist()
#     ics = pred[param_dim : param_dim + ic_dim].tolist()
#     forecasts = pred[param_dim + ic_dim : param_dim + ic_dim + forecast_dim].tolist()
#     return {
#         "parameters": params,
#         "initial_conditions": ics,
#         "forecast_values": forecasts,
#     }


def predict_fn(
    input_data: list[dict[str, Any]], model: CNN1DResidualPredictor
) -> list[list[float]]:
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for sample in input_data:
            x, x_feat = _prepare_tensor(sample, model.config)
            x, x_feat = x.to(device), x_feat.to(device)
            pred = model(x, x_feat).squeeze().cpu().numpy()
            outputs.append(pred.tolist())
    return outputs


def output_fn(prediction: list[list[float]], accept: str) -> tuple[str, str]:
    if accept not in {"application/json", None}:
        raise ValueError("Only application/json is supported for output")
    return json.dumps(prediction), "application/json"
