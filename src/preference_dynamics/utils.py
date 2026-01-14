from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import chain, combinations
from pathlib import Path
from typing import Any, TypeVar, cast

import mlflow
import torch
from optuna.trial import FixedTrial, FrozenTrial, Trial

type TrialLike = Trial | FrozenTrial | FixedTrial

T = TypeVar("T")


def num_vars(n: int) -> int:
    """
    Compute number of variables for n actions.
    """
    return 2 * n


def num_params(n: int) -> int:
    """
    Compute number of parameters for n actions.
    """
    return 2 * n + 2 * n**2


def get_var_names(n: int, suffix: str | None = None) -> list[str]:
    """
    Get names of variables for n actions.
    """
    suffix_str = "" if suffix is None else f"_{suffix}"
    return [f"u_{i}{suffix_str}" for i in range(n)] + [f"a_{i}{suffix_str}" for i in range(n)]


def get_param_names(n: int, ic: bool = False, prefix: str = "") -> list[str]:
    """
    Compute names of parameters for n actions.
    """

    n_range = range(n)
    param_names = (
        [f"g_{i}" for i in n_range]
        + [f"μ_{i}" for i in n_range]
        + [f"Π_{i},{j}" for i in n_range for j in n_range]
        + [f"Γ_{i},{j}" for i in n_range for j in n_range]
        + [f"v0_{i}" for i in n_range if ic]
        + [f"m0_{i}" for i in n_range if ic]
    )
    param_names = [f"{prefix}{name}" for name in param_names]
    return param_names


def get_diagonal_indices(n: int) -> list[int]:
    """
    Compute indices of diagonal elements in parameter vector.

    The parameter vector structure is: [g, μ, Π, Γ]
    - g: [0:n]
    - μ: [n:2n]
    - Π: [2n:2n+n²] (matrix in row-major order)
    - Γ: [2n+n²:2n+2n²] (matrix in row-major order)

    Diagonal constraints:
    - Π[i,i] > 0 for i=0,...,n-1 (indices: 2n + i*n + i)
    - Γ[i,i] > 0 for i=0,...,n-1 (indices: 2n + n² + i*n + i)

    Args:
        n: Number of actions (n)

    Returns:
        List of indices for Π and Γ diagonal elements that must be positive

    Example:
        >>> get_diagonal_indices(3)
        [6, 10, 14, 15, 19, 23]  # Π diagonals + Γ diagonals
    """
    diagonal_indices = []

    # Π diagonal indices: 2n + i*n + i for i=0,...,n-1
    for i in range(n):
        diagonal_indices.append(2 * n + i * n + i)

    # Γ diagonal indices: 2n + n² + i*n + i for i=0,...,n-1
    for i in range(n):
        diagonal_indices.append(2 * n + n**2 + i * n + i)

    return diagonal_indices


def get_subsets(n: int | Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """
    Generate all subsets of an iterable or range(n).

    Args:
        n: Integer (generates subsets of range(n)) or iterable

    Returns:
        Iterator over all subsets as tuples, including empty set

    Example:
        >>> list(get_subsets(3))
        [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        >>> list(get_subsets([1, 3]))
        [(), (1,), (3,), (1, 3)]
    """

    def _get_subsets(lst: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
        return chain.from_iterable(combinations(lst, r) for r in range(len(lst) + 1))  # type: ignore

    if isinstance(n, int):
        return _get_subsets(range(n))
    else:
        return _get_subsets(n)


def add_prefix(params: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    Add prefix to dictionary keys.
    """
    return {f"{prefix}_{k}": v for k, v in params.items()}


def if_logging[T](fn: Callable[..., T]) -> Callable[..., T | None]:
    """
    Wrap a function to execute only in an active MLflow run.
    Can be used as a decorator.
    """

    def wrapper(*args: Any, **kwargs: Any) -> T | None:
        return fn(*args, **kwargs) if mlflow.active_run() else None

    return wrapper


def ensure_trial_instance(trial: TrialLike | dict[str, Any] | None) -> TrialLike:
    """
    Convert dict/None to FixedTrial, pass through Trial objects.

    Args:
        trial: TrialLike (FixedTrial, Trial, or FrozenTrial), dict of params, or None

    Returns:
        TrialLike object

    Raises:
        TypeError: If trial is invalid type
    """
    if isinstance(trial, (Trial, FrozenTrial, FixedTrial)):
        return trial
    elif isinstance(trial, dict) or trial is None:
        return FixedTrial(params=trial or {})
    raise TypeError(f"Invalid trial type: {type(trial)}")


def parse_checkpoint_path(path: str | Path) -> list[str]:
    """
    Parse checkpoint path of the form (file extension `.pt` is enforced):
    {checkpoint_dir}/{model_name}/{run_id}/{filename}
    {model_name}/{run_id}/{filename}
    {run_id}/{filename}
    {filename}

    Returns list of parts: (checkpoint_dir, model_name, run_id, filename). Missing parts are dropped.
    """
    p = Path(path).with_suffix(".pt")
    parts = list(p.parts)
    if len(parts) > 4:
        raise ValueError(f"Invalid checkpoint path: {path}. Too many subdirectories (>3).")

    return parts


def assemble_checkpoint_path(
    parts: Sequence[str],
    *,
    checkpoint_dir: str = "./checkpoints",
    model_name: str | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Assemble checkpoint path from the provided parts:
    [checkpoint_dir, model_name, run_id, filename.pt]
    [model_name, run_id, filename.pt]
    [run_id, filename.pt]
    [filename.pt]
    If any part is missing, it will be inferred from the kwargs.

    Args:
        parts: List of parts to assemble.
        checkpoint_dir: Checkpoint directory.
        model_name: Model name.
        run_id: Run ID.

    Returns:
        Path to checkpoint.
    """
    out = list(parts)
    length = len(out)

    if length == 0:
        raise ValueError("No parts provided.")
    if length == 1:
        if run_id is None:
            raise ValueError("Run ID is required for single part path.")
        out = [run_id] + out
    if length <= 2:
        if model_name is None:
            raise ValueError("Model name is required for two part path.")
        out = [model_name] + out
    if length <= 3:
        out = [checkpoint_dir] + out
    if length > 4:
        raise ValueError("Too many subdirectories provided.")

    return Path(*out)


def to_device[T](data: T, device: torch.device) -> T:
    """
    Move all tensors in data to device. Supports tuple, list, dict.
    """
    if isinstance(data, torch.Tensor):
        out = data.to(device)
    elif isinstance(data, dict):
        out = {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, tuple):
        out = tuple(to_device(v, device) for v in data)
    elif isinstance(data, list):
        out = [to_device(v, device) for v in data]
    else:
        out = data
    return cast(T, out)


def to_cpu_numpy[T](data: T) -> T:
    """
    Move all tensors in data to CPU and convert to numpy. Supports tuple, list, dict.
    """
    if isinstance(data, torch.Tensor):
        out = data.cpu().numpy()
    elif isinstance(data, dict):
        out = {k: to_cpu_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        out = tuple(to_cpu_numpy(v) for v in data)
    elif isinstance(data, list):
        out = [to_cpu_numpy(v) for v in data]
    else:
        out = data
    return cast(T, out)


def stack_dict_tensors(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Stacks tensors in a list of dictionaries into a single dictionary of stacked tensors.
    Assumes all dictionaries have the same keys.
    """
    out: dict[str, list[torch.Tensor]] = {k: [] for k in data[0]}
    for pred in data:
        for key, value in pred.items():
            out[key].append(value)

    out_tensor = {k: torch.vstack(v) for k, v in out.items()}
    return out_tensor


def to_mathematica_parameters(config) -> None:  # type: ignore
    """
    Print ODEConfig as Mathematica parameter assignment string, for debugging purposes.

    Converts ODEConfig to Mathematica format and prints the result.
    Compatible with motivation_dynamics_4_pref.nb

    Args:
        config: ODEConfig object with parameters and initial conditions

    Returns:
        None (prints result to stdout)
    """
    n: int = config.n_actions
    parts: list[str] = []

    # g values -> \[CurlyPhi]01, \[CurlyPhi]02, etc. (1-indexed)
    g = config.parameters.g
    for i in range(n):
        parts.append(f"\\[CurlyPhi]0{i + 1} -> {g[i]}")

    # mu values -> m1, m2, etc. (1-indexed)
    mu = config.parameters.mu
    for i in range(n):
        parts.append(f"m{i + 1} -> {mu[i]}")

    # Pi matrix (row-major, 1-indexed)
    Pi = config.parameters.Pi
    for i in range(n):
        for j in range(n):
            parts.append(f"\\[Pi]{i + 1}{j + 1} -> {Pi[i, j]}")

    # Gamma matrix (row-major, 1-indexed)
    Gamma = config.parameters.Gamma
    for i in range(n):
        for j in range(n):
            parts.append(f"\\[Gamma]{i + 1}{j + 1} -> {Gamma[i, j]}")

    # Initial conditions: first n values -> u01, u02, etc.
    # Last n values -> a01, a02, etc.
    v = config.initial_conditions.v
    for i in range(n):
        parts.append(f"u0{i + 1} -> {v[i]}")
    m = config.initial_conditions.m
    for i in range(n):
        parts.append(f"a0{i + 1} -> {m[i]}")

    # Format as Mathematica assignment
    result = "valScenario = {" + ", ".join(parts) + "};"
    print(result)
