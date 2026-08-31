"""Validation for frozen, development-fitted gate-threshold artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gate_thresholds(
    artifact_path: Path,
    *,
    config_path: Path,
    reliability_checkpoint: Path,
    generator_checkpoint: Path,
    generator_kind: str = "flow",
) -> tuple[float, float, dict]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("role") != "development_gate_calibration_with_hash_disjoint_audit":
        raise ValueError("Unsupported gate-calibration role")
    expected_control = {
        "flow": "fixed_seed_k1",
        "deterministic": "deterministic_point_estimate",
    }.get(generator_kind)
    if expected_control is None:
        raise ValueError(f"Unsupported gate generator kind: {generator_kind}")
    if payload.get("selection_control") != expected_control:
        raise ValueError(
            "Gate calibration selection control/generator kind mismatch"
        )
    declared_kind = payload.get("generator_kind", "flow")
    if declared_kind != generator_kind:
        raise ValueError("Gate calibration generator kind mismatch")
    if payload.get("split_unit") != "source_group":
        raise ValueError("Gate calibration must use source-group-disjoint folds")
    expected = {
        "config_sha256": sha256(config_path),
        "reliability_checkpoint_sha256": sha256(reliability_checkpoint),
        "flow_checkpoint_sha256": sha256(generator_checkpoint),
    }
    for field, actual in expected.items():
        if payload.get(field) != actual:
            raise ValueError(f"Gate-calibration {field} mismatch")
    best = payload.get("best_fit", {})
    low = float(best.get("tau_low", float("nan")))
    high = float(best.get("tau_high", float("nan")))
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Invalid calibrated gate thresholds")
    return low, high, payload
