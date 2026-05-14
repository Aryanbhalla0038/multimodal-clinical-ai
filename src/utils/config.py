"""YAML config loader with simple dot-access."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


class Config(dict):
    """Dict that also supports attribute access recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_yaml(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)
