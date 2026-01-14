import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_config(model_dir: Path) -> dict[str, Any]:
    """
    Load model config from disk.

    Args:
        model_dir: Directory containing the model.

    Returns:
        Model config.
    """
    config_path = model_dir / "model_config.json"
    config: dict[str, Any] = json.loads(config_path.read_text())
    return config


def _load_model(path: str | Path) -> torch.nn.Module:
    """
    Load scripted model from disk for inference.

    Args:
        path: File path to load model from
    """
    model = torch.jit.load(path)
    return model


def model_fn(model_dir: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_path = Path(model_dir)
    config = _load_config(model_path)
    model = _load_model(model_path / "model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"MODEL_FN: model loaded from {model_path / 'model.pt'}")
    return (model, config)


def _ensure_list_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert dict to list of dicts if not already a list.

    Args:
        payload: Payload to convert

    Returns:
        List of dicts.
    """
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return payload
        raise ValueError("JSON array must contain dicts")
    raise ValueError("Unsupported JSON payload; expected dict or array of dicts")


def input_fn(request_body: str, content_type: str) -> list[dict[str, Any]]:
    # input_fn requires JSONSerializer for deployment to circumvent SageMaker utf conversion bug
    if content_type != "application/json":
        raise ValueError("Only ContentType=application/json is supported for input")
    payload = json.loads(request_body)
    payload = _ensure_list_payload(payload)
    return payload


def _prepare_tensor(
    sample: dict[str, Any],
    in_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sample to input tensors for residual CNN model.

    Args:
        sample: Sample to prepare
        in_channels: Number of input channels

    Returns:
        Tuple of tensors.
    """
    x_np = np.asarray(sample["time_series"])
    x_feat_np = np.asarray(sample["features"])

    if x_np.shape[0] != in_channels:
        raise ValueError(f"Trained model got {x_np.shape[0]} channels, expected {in_channels}.")

    x = torch.from_numpy(x_np).float()
    x_feat = torch.from_numpy(x_feat_np).float()
    return x, x_feat


def predict_fn(
    input_data: list[dict[str, Any]], model_tuple: tuple[torch.nn.Module, dict[str, Any]]
) -> list[list[float]]:
    model, config = model_tuple
    device = next(model.parameters()).device
    outputs = []
    with torch.inference_mode():
        # TODO: optimize for batch inference
        for sample in input_data:
            x, x_feat = _prepare_tensor(sample, config["in_channels"])
            x, x_feat = x.to(device), x_feat.to(device)
            pred = model(x, x_feat).squeeze().cpu().numpy()
            outputs.append(pred.tolist())
    logger.info(f"PREDICT_FN: num predictions: {len(outputs)}")
    return outputs


def output_fn(prediction: list[list[float]], accept: str) -> tuple[str, str]:
    if accept not in {"application/json", None}:
        raise ValueError("Only Accept=application/json is supported for output")
    return json.dumps(prediction), "application/json"
