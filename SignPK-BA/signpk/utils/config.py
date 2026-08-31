from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and its simple ``defaults`` includes.

    Include paths are resolved relative to the including file. Includes are
    merged first and the current file wins. This intentionally small loader
    avoids making Hydra a runtime requirement for inference/evaluation.
    """

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    defaults = current.pop("defaults", {}) or {}
    merged: dict[str, Any] = {}
    if isinstance(defaults, list):
        entries = {str(i): value for i, value in enumerate(defaults)}
    elif isinstance(defaults, Mapping):
        entries = defaults
    else:
        raise TypeError("config defaults must be a mapping or list")
    for _, include in entries.items():
        include_path = (path.parent / str(include)).resolve()
        merged = deep_merge(merged, load_yaml(include_path))
    return deep_merge(merged, current)


def project_path(value: str | Path, project_root: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()

