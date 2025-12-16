"""
IO handlers for saving and loading data.
"""

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from preference_dynamics.schemas import TimeSeriesSample


class IOHandler(Protocol):
    """
    IO handler for saving and loading data.
    """

    suffix: str

    def save(self, data: Any, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> Any: ...


class PickleHandler:
    """
    Handler for saving and loading data as pickle files.
    Always uses suffix ".pkl" (regardless of the provided suffix).
    """

    suffix = ".pkl"

    def save(self, data: Any, path: str | Path) -> None:
        path = Path(path).with_suffix(self.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> Any:
        path = Path(path).with_suffix(self.suffix)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("rb") as f:
            return pickle.load(f)


class JSONHandler:
    """
    Handler for saving and loading data as JSON files.
    Always uses suffix ".json" (regardless of the provided suffix).
    """

    suffix = ".json"

    def save(self, data: Sequence[TimeSeriesSample], path: str | Path) -> None:
        path = Path(path).with_suffix(self.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)

        data_dicts = [item.model_dump_json() for item in data]

        with path.open("w", encoding="utf-8") as f:
            json.dump(data_dicts, f)

    def load(self, path: str | Path) -> Sequence[TimeSeriesSample]:
        path = Path(path).with_suffix(self.suffix)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data_dicts = json.load(f)
            return [TimeSeriesSample.model_validate_json(item) for item in data_dicts]
