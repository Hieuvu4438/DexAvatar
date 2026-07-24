"""Configuration helpers shared by Phase 2 commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def nested_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def validate_config(
    config: dict[str, Any],
    require_data: bool = False,
    require_validation: bool = False,
) -> None:
    from phase2_refiner.data.dataset import TOKEN_FEATURE_DIM

    model = config.get("model", {})
    input_dim = int(model.get("input_dim", TOKEN_FEATURE_DIM))
    if input_dim != TOKEN_FEATURE_DIM:
        raise ValueError(
            f"model.input_dim={input_dim}, but cache-v2 tokens require {TOKEN_FEATURE_DIM}"
        )
    hidden = int(model.get("hidden_size", 256))
    heads = int(model.get("num_heads", 8))
    if hidden % heads:
        raise ValueError("model.hidden_size must be divisible by model.num_heads")
    if int(model.get("max_frames", 64)) < 2:
        raise ValueError("model.max_frames must be at least 2")
    if require_data:
        data = config.get("data", {})
        if not data.get("train_glob"):
            raise ValueError("Missing data.train_glob")
        if require_validation and not data.get("val_glob"):
            raise ValueError("Missing data.val_glob")
