"""Resolve local or server data roots without mixing their absolute paths."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


def resolve_data_paths(config_path: str | Path, environment: str) -> dict[str, Path]:
    """Return absolute paths derived only from the requested environment root."""

    if environment not in {"local", "server"}:
        raise ValueError("environment must be 'local' or 'server'.")
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or not isinstance(config.get("data"), Mapping):
        raise ValueError("Deployment path config requires a data mapping.")
    data = config["data"]
    root = data.get(f"{environment}_data_root")
    derived = data.get("derived")
    if not isinstance(root, str) or not Path(root).is_absolute() or not isinstance(derived, Mapping):
        raise ValueError("Deployment path config requires absolute environment roots and a derived mapping.")
    result: dict[str, Path] = {}
    for name, relative in derived.items():
        if not isinstance(name, str) or not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Derived deployment paths must be relative children of their data root.")
        result[name] = Path(root) / relative
    return result
