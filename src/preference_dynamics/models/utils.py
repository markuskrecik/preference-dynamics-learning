from pathlib import Path

import torch


def num_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: torch.nn.Module,
    path: str | Path,
    trace: bool = False,
    example_kwarg_inputs: dict[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    """
    Save model torchscript to disk.

    Args:
        model: PyTorch model to save
        path: File path to save model (e.g., 'model.pt')
        trace: Whether to trace the model
        example_kwarg_inputs: Example keyword arguments to trace the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_training = model.training
    model.eval()
    if trace:
        if example_kwarg_inputs is None:
            raise ValueError("example_kwarg_inputs must be provided when tracing the model")
        with torch.no_grad():
            model(**example_kwarg_inputs)
        scripted_model = torch.jit.trace(model, example_kwarg_inputs=example_kwarg_inputs)
    else:
        scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, path)
    model.train(is_training)


def load_model(path: str | Path) -> torch.nn.Module:
    """
    Load scripted model from disk for inference.

    Args:
        path: File path to load model from
    """
    model = torch.jit.load(path)
    return model
