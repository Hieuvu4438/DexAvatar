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
    from phase2_refiner.data.dataset import (
        TOKEN_FEATURE_DIM,
        TOKEN_FEATURE_DIM_WITH_REPROJECTION,
    )

    model = config.get("model", {})
    data = config.get("data", {})
    input_dim = int(model.get("input_dim", TOKEN_FEATURE_DIM))
    if input_dim not in (TOKEN_FEATURE_DIM, TOKEN_FEATURE_DIM_WITH_REPROJECTION):
        raise ValueError(
            f"model.input_dim={input_dim}, but supported token layouts are "
            f"{TOKEN_FEATURE_DIM} and {TOKEN_FEATURE_DIM_WITH_REPROJECTION}"
        )
    if (
        model.get("use_reprojection_skip", False)
        and input_dim != TOKEN_FEATURE_DIM_WITH_REPROJECTION
    ):
        raise ValueError("model.use_reprojection_skip requires the 45-feature layout")
    if model.get("uncertainty_feedback", False) and not model.get(
        "predict_uncertainty", False
    ):
        raise ValueError(
            "model.uncertainty_feedback requires model.predict_uncertainty"
        )
    if float(config.get("loss", {}).get("benefit_weight", 0.0)) > 0 and not model.get(
        "predict_benefit", False
    ):
        raise ValueError("loss.benefit_weight requires model.predict_benefit")
    if config.get("loss", {}).get("target_quality_weighting", False) and not data.get(
        "require_phase2r_semantics", False
    ):
        raise ValueError(
            "loss.target_quality_weighting requires data.require_phase2r_semantics"
        )
    if data.get("formal_evidence", False):
        if not data.get("require_phase2r_semantics", False):
            raise ValueError(
                "data.formal_evidence requires data.require_phase2r_semantics"
            )
        if not data.get("formal_contract_report"):
            raise ValueError(
                "data.formal_evidence requires data.formal_contract_report"
            )
    threshold = config.get("inference", {}).get("benefit_threshold")
    if threshold is not None:
        if not model.get("predict_benefit", False):
            raise ValueError(
                "inference.benefit_threshold requires model.predict_benefit"
            )
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("inference.benefit_threshold must be within (0, 1)")
    residual_scale = float(data.get("reprojection_residual_scale", 10.0))
    if residual_scale <= 0:
        raise ValueError("data.reprojection_residual_scale must be positive")
    reference_seconds = float(data.get("motion_reference_seconds", 0.04))
    if reference_seconds <= 0:
        raise ValueError("data.motion_reference_seconds must be positive")
    loss = config.get("loss", {})
    benefit_target = str(loss.get("benefit_target", "rotation"))
    if benefit_target not in {"rotation", "vertex"}:
        raise ValueError("loss.benefit_target must be 'rotation' or 'vertex'")
    if benefit_target == "vertex" and not config.get("geometry", {}).get(
        "enabled", False
    ):
        raise ValueError("loss.benefit_target=vertex requires geometry.enabled")
    if "physical_time_motion" in loss and bool(loss["physical_time_motion"]) != bool(
        data.get("physical_time_motion", False)
    ):
        raise ValueError(
            "loss.physical_time_motion must match data.physical_time_motion"
        )
    if (
        "motion_reference_seconds" in loss
        and float(loss["motion_reference_seconds"]) != reference_seconds
    ):
        raise ValueError(
            "loss.motion_reference_seconds must match data.motion_reference_seconds"
        )
    hidden = int(model.get("hidden_size", 256))
    heads = int(model.get("num_heads", 8))
    if hidden % heads:
        raise ValueError("model.hidden_size must be divisible by model.num_heads")
    if int(model.get("max_frames", 64)) < 2:
        raise ValueError("model.max_frames must be at least 2")
    checkpoint_metric = str(
        config.get("training", {}).get("checkpoint_metric", "rotation")
    )
    if checkpoint_metric not in {"rotation", "vertex"}:
        raise ValueError("training.checkpoint_metric must be 'rotation' or 'vertex'")
    if checkpoint_metric == "vertex" and not config.get("geometry", {}).get(
        "enabled", False
    ):
        raise ValueError("training.checkpoint_metric=vertex requires geometry.enabled")
    checkpoint_validation = str(
        config.get("training", {}).get("checkpoint_validation", "corrupted")
    )
    if checkpoint_validation not in {"clean", "corrupted"}:
        raise ValueError(
            "training.checkpoint_validation must be 'clean' or 'corrupted'"
        )
    from phase2_refiner.t5_optimize import validate_t5_config

    validate_t5_config(config.get("t5", {}))
    if require_data:
        if not data.get("train_glob"):
            raise ValueError("Missing data.train_glob")
        if require_validation and not data.get("val_glob"):
            raise ValueError("Missing data.val_glob")
