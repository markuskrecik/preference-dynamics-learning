"""
Unit tests for utils.py.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from optuna import create_study
from optuna.trial import FixedTrial

from preference_dynamics.utils import (
    assemble_checkpoint_path,
    ensure_trial_instance,
    get_diagonal_indices,
    get_subsets,
    if_logging,
    parse_checkpoint_path,
)


class TestGetDiagonalIndices:
    """Test suite for get_diagonal_indices helper function."""

    # TODO: parametrize across n=1,2,3
    def test_n1_diagonal_indices(self) -> None:
        """Test diagonal indices for n=1 actions."""
        indices = get_diagonal_indices(1)
        assert indices == [2, 3]

    def test_n2_diagonal_indices(self) -> None:
        """Test diagonal indices for n=2 actions."""
        indices = get_diagonal_indices(2)
        assert indices == [4, 7, 8, 11]

    def test_n3_diagonal_indices(self) -> None:
        """Test diagonal indices for n=3 actions."""
        indices = get_diagonal_indices(3)
        assert indices == [6, 10, 14, 15, 19, 23]

    # TODO: put assertion into tests above
    def test_diagonal_indices_length(self) -> None:
        """Test that we get 2n diagonal indices for n actions."""
        for n in range(1, 6):
            indices = get_diagonal_indices(n)
            assert len(indices) == 2 * n

    def test_diagonal_indices_within_bounds(self) -> None:
        """Test that all indices are within parameter vector bounds."""
        for n in range(1, 6):
            indices = get_diagonal_indices(n)
            param_length = 2 * n + 2 * n**2
            assert all(0 <= idx < param_length for idx in indices)


class TestEnsureTrialInstance:
    def test_ensure_trial_instance_with_none(self) -> None:
        """Test that ensure_trial_instance() handles None."""
        trial = ensure_trial_instance(None)
        assert isinstance(trial, FixedTrial)
        assert trial.params == {}

    def test_ensure_trial_instance_with_dict(self, tmp_path: Path) -> None:
        """Test that ensure_trial_instance() handles dict."""
        trial = ensure_trial_instance({"param1": 1.0, "param2": 2.0})
        assert isinstance(trial, FixedTrial)
        # FixedTrial.params yields {}, bug in Optuna?
        assert trial._params == {"param1": 1.0, "param2": 2.0}

    def test_ensure_trial_instance_with_trial(self) -> None:
        """Test that ensure_trial_instance() passes through Trial objects."""
        study = create_study()
        trial = study.ask()
        assert ensure_trial_instance(trial) is trial


class TestParseCheckpointPath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (
                Path("checkpoints_dir/model_name/run_id/filename.pt"),
                ["checkpoints_dir", "model_name", "run_id", "filename.pt"],
            ),
            ("model_name/run_id/filename.pt", ["model_name", "run_id", "filename.pt"]),
            ("run_id/filename", ["run_id", "filename.pt"]),
            ("filename", ["filename.pt"]),
            (
                "checkpoints_dir/model_name/run_id/filename",
                ["checkpoints_dir", "model_name", "run_id", "filename.pt"],
            ),
            ("model_name/run_id/filename", ["model_name", "run_id", "filename.pt"]),
            (Path("run_id/filename"), ["run_id", "filename.pt"]),
            (Path("filename"), ["filename.pt"]),
        ],
    )
    def test_parse_checkpoint_path_valid(self, path: str | Path, expected: list[str]) -> None:
        """Test that parse_checkpoint_path() correctly parses valid checkpoint paths."""
        result = parse_checkpoint_path(path)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_path",
        [
            "a/b/c/d/filename",
            "checkpoints_dir/model_name/run_id/subdir/filename.pt",
            Path("a/b/c/d/filename.pt"),
        ],
    )
    def test_parse_checkpoint_path_too_many_parts(self, invalid_path: str | Path) -> None:
        """Test that parse_checkpoint_path() raises ValueError for paths with too many parts."""
        with pytest.raises(ValueError, match="Too many subdirectories"):
            parse_checkpoint_path(invalid_path)

    def test_parse_checkpoint_path_empty_string(self) -> None:
        """Test that parse_checkpoint_path() raises ValueError for empty string."""
        with pytest.raises(ValueError, match="has an empty name"):
            parse_checkpoint_path("")

    def test_parse_checkpoint_path_existing_suffix(self) -> None:
        """Test that parse_checkpoint_path() replaces existing suffix with .pt."""
        result = parse_checkpoint_path("filename.pth")
        assert result == ["filename.pt"]

    def test_parse_checkpoint_path_multiple_dots(self) -> None:
        """Test that parse_checkpoint_path() replaces suffix (everything after last dot) with .pt."""
        result = parse_checkpoint_path("model.name.v2")
        assert result == ["model.name.pt"]


class TestAssembleCheckpointPath:
    @pytest.mark.parametrize(
        ("parts", "kwargs", "expected"),
        [
            (["run", "model", "file.pt"], {}, Path("./checkpoints/run/model/file.pt")),
            (
                ["file.pt"],
                {"run_id": "runX", "model_name": "m"},
                Path("./checkpoints/m/runX/file.pt"),
            ),
        ],
    )
    def test_assemble_checkpoint_path(
        self, parts: list[str], kwargs: dict[str, str], expected: Path
    ) -> None:
        assert assemble_checkpoint_path(parts, **kwargs) == expected

    def test_assemble_checkpoint_path_rejects_missing_parts(self) -> None:
        with pytest.raises(ValueError):
            assemble_checkpoint_path([], run_id="a")


def test_get_subsets_returns_expected_combinations() -> None:
    subsets = list(get_subsets(3))
    expected = [
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    assert subsets == expected


def test_if_logging_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int] = []

    def _record(value: int) -> int:
        captured.append(value)
        return value

    wrapped = if_logging(_record)
    monkeypatch.setattr("mlflow.active_run", lambda: None)

    assert wrapped(1) is None

    monkeypatch.setattr("mlflow.active_run", lambda: SimpleNamespace(info=None))
    assert wrapped(2) == 2
    assert captured == [2]
