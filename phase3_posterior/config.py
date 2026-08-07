"""Configuration loading and fail-closed validation for Phase 3."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("data", "model", "diffusion", "training")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML with an optional relative ``base`` configuration."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    base_name = payload.pop("base", None)
    if base_name is not None:
        base_path = (config_path.parent / str(base_name)).resolve()
        payload = _merge(load_config(base_path), payload)
    payload["_config_path"] = str(config_path)
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    model = config["model"]
    diffusion = config["diffusion"]
    training = config["training"]
    if int(model.get("num_joints", 51)) != 51:
        raise ValueError("Phase 3 requires exactly 51 body/hand joints")
    if int(model.get("max_frames", 64)) < 8:
        raise ValueError("model.max_frames must be at least 8")
    if float(diffusion.get("beta_min", 0.1)) <= 0:
        raise ValueError("diffusion.beta_min must be positive")
    if float(diffusion.get("beta_max", 20.0)) <= float(diffusion.get("beta_min", 0.1)):
        raise ValueError("diffusion.beta_max must exceed beta_min")
    if not 0 < float(diffusion.get("eps", 1e-3)) < 1:
        raise ValueError("diffusion.eps must be in (0, 1)")
    if int(training.get("workers", 0)) > 4:
        raise ValueError("Phase 3 CPU worker cap is 4")
    if int(training.get("gradient_accumulation", 1)) < 1:
        raise ValueError("training.gradient_accumulation must be positive")
    if int(training.get("early_stop_patience", 0)) < 0:
        raise ValueError("training.early_stop_patience cannot be negative")
    if int(training.get("hint_only_steps", 0)) < 0:
        raise ValueError("training.hint_only_steps cannot be negative")
    if training.get("conditioning_validity", "frame_valid_initializer") != (
        "frame_valid_initializer"
    ):
        raise ValueError(
            "Phase 3 masked diffusion requires frame-valid initializer conditioning"
        )
    validation_interval = int(training.get("validation_interval", 0))
    if validation_interval < 0:
        raise ValueError("training.validation_interval cannot be negative")
    if validation_interval > 0:
        if not config["data"].get("val_index"):
            raise ValueError("training validation requires data.val_index")
        for key in (
            "validation_batch_size",
            "validation_max_batches",
            "validation_sampling_steps",
        ):
            if int(training.get(key, 1)) < 1:
                raise ValueError(f"training.{key} must be positive")
    if model.get("masked_rotation_hints", False):
        if float(training.get("masked_rotation_corruption_degrees", 0.0)) <= 0:
            raise ValueError(
                "masked rotation hints require positive corruption degrees"
            )
        if model.get("reset_conditioning_projections_on_init", False):
            raise ValueError(
                "conditional warm starts must not reset trained conditioning"
            )
    fallback = config.get("fallback", {})
    if fallback.get("mode") == "geometry_only":
        required_false = (
            "contact_energy_enabled",
            "force_coupling_enabled",
            "persistence_constraints_enabled",
        )
        for key in required_false:
            if fallback.get(key) is not False:
                raise ValueError(f"geometry-only fallback requires fallback.{key}=false")
        if model.get("contact_energy_enabled") is not False:
            raise ValueError(
                "geometry-only fallback requires model.contact_energy_enabled=false"
            )
        loss = config.get("loss", {})
        for key in ("contact", "persistence"):
            if float(loss.get(key, 0.0)) != 0.0:
                raise ValueError(f"geometry-only fallback requires loss.{key}=0")


def save_resolved_config(config: dict[str, Any], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)
