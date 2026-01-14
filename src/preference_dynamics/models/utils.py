from pathlib import Path
from typing import Any, Literal

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


def get_jacobian_diagonal(
    f: torch.nn.Module,
    t: torch.Tensor,
    *args: Any,
    mode: Literal["reverse", "forward"] = "reverse",
) -> torch.Tensor:
    """
    Compute ∂f[...,i,j]/∂t[...,j] for all i,j

    Args:
        f: Differentiable function
            must handle single (n,) -> (m, n), and batched inputs (B, n) -> (B, m, n)
        t: input tensor, shape (n,) or (B, n)
        *args: Additional non-differentiated arguments
        mode: "forward" for `jacfwd`, "reverse" for `jacrev` (default: "reverse")

    Returns:
        diagonal: tensor where diagonal[...,i,j] = ∂f[...,i,j]/∂t[...,j]
    """
    is_batched = t.ndim == 2

    if mode == "reverse":
        jac_fn = torch.func.jacrev
    elif mode == "forward":
        jac_fn = torch.func.jacfwd
    else:
        raise ValueError("Unsupported mode. Supported modes are: 'reverse', 'forward'.")

    def jacobian_diag_single(t_single: torch.Tensor, *args: Any) -> torch.Tensor:
        """Compute diagonal for a single sample (no batch)"""
        jac = jac_fn(f)(t_single, *args)  # (m, n, n)
        n = jac.shape[-1]
        return jac[..., torch.arange(n), torch.arange(n)]  # (m, n)

    if is_batched:
        in_dims = (0,) * (1 + len(args))
        return torch.func.vmap(jacobian_diag_single, in_dims=in_dims, randomness="same")(
            t, *args
        )  # (B, m, n)
    else:
        return jacobian_diag_single(t, *args)  # (m, n)


def get_jacobian_diagonal_autograd(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if x.ndim != t.ndim + 1:
        raise ValueError(
            f"Dimension mismatch, expected x.ndim == t.ndim + 1, got {x.ndim} != {t.ndim} + 1."
        )
    if not t.requires_grad:
        raise ValueError("t.requires_grad must be True.")

    def compute_grad_single(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        grad = torch.zeros_like(x)
        for i in range(m):
            for j in range(n):
                grad_output = torch.zeros_like(x)
                grad_output[i, j] = 1.0
                grad_t = torch.autograd.grad(
                    x, t, grad_outputs=grad_output, create_graph=True, retain_graph=True
                )[0]
                grad[i, j] = grad_t[j] if grad_t is not None else 0.0
        return grad

    is_batched = t.ndim == 2
    if not is_batched:
        grad = compute_grad_single(x, t)
    else:
        # B = x.shape[0]
        # grad_lst = [compute_grad_single(x[b], t[b]) for b in range(B)]
        # grad = torch.stack(grad_lst)

        B, m, n = x.shape
        grad = torch.zeros_like(x)
        for b in range(B):
            for i in range(m):
                for j in range(n):
                    grad_output = torch.zeros_like(x)
                    grad_output[b, i, j] = 1.0
                    grad_t = torch.autograd.grad(
                        x,
                        t,
                        grad_outputs=grad_output,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    grad[b, i, j] = grad_t[b, j] if grad_t is not None else 0.0
        return grad
    return grad
