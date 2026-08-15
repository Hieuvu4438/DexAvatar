"""Fail-closed, machine-readable audit of the complete Phase 2 gate chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from phase2_refiner.data.materialize_signavatars_targets import (
    validate_license_acceptance,
    validate_target_audit,
)


SCHEMA = "phase2-full-go-audit-v1"
REGIONS = ("ubody", "lhand", "rhand")
CAUSAL_G5_CHECKS = (
    "feedback_intervention_improves_corrupt_reconstruction",
    "feedback_intervention_clean_regression_at_most_1pct",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _all_true(mapping: Any) -> bool:
    return isinstance(mapping, dict) and bool(mapping) and all(
        value is True for value in mapping.values()
    )


def _status(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    value = path.read_text(encoding="utf-8").strip()
    return value or "EMPTY"


def _g0_passed(g0: dict[str, Any]) -> bool:
    prediction = g0.get("prediction")
    return (
        int(g0.get("frames", 0)) == 1493
        and int(g0.get("signs", 0)) == 57
        and _is_sha256(g0.get("manifest_sha256"))
        and isinstance(prediction, dict)
        and all(float(prediction.get(region, float("inf"))) > 0 for region in REGIONS)
    )


def _g1_passed(g0: dict[str, Any], g1: dict[str, Any]) -> bool:
    bootstrap = g1.get("paired_bootstrap")
    prediction = g1.get("prediction")
    baseline = g1.get("baseline")
    return (
        g1.get("manifest_sha256") == g0.get("manifest_sha256")
        and int(g1.get("frames", 0)) == int(g0.get("frames", -1))
        and int(g1.get("signs", 0)) == int(g0.get("signs", -1))
        and isinstance(bootstrap, dict)
        and isinstance(prediction, dict)
        and isinstance(baseline, dict)
        and all(
            float(prediction.get(region, float("inf")))
            <= float(baseline.get(region, float("-inf")))
            and float(bootstrap.get(region, {}).get("mean_delta_mm", float("inf")))
            < 0
            and float(bootstrap.get(region, {}).get("ci95_high_mm", float("inf")))
            < 0
            for region in REGIONS
        )
    )


def _historical_g2_core_passed(g2: dict[str, Any]) -> bool:
    train = g2.get("train", {})
    volume = g2.get("sign_domain_training_volume", {})
    enough_volume = int(volume.get("clips", 0)) >= 10_000 or int(
        volume.get("frames", 0)
    ) >= 250_000
    return (
        _all_true(g2.get("gates"))
        and enough_volume
        and float(train.get("fraction_clips_at_least_16", 0.0)) >= 0.80
        and float(train.get("complete_body_and_both_hand_fraction", 0.0)) >= 0.70
        and not g2.get("train_validation_clip_overlap")
        and not g2.get("train_validation_source_group_overlap")
    )


def _g3_passed(g3: dict[str, Any]) -> bool:
    gates = g3.get("gates", {})
    return (
        gates.get("G3") is True
        and gates.get("regional_recovery_at_least_30_percent") is True
        and gates.get("clean_to_injected_below_2_percent") is True
        and g3.get("translation_centered_per_region") is True
    )


def _g5_passed(g5: dict[str, Any], formal_passed: bool) -> bool:
    gate = g5.get("gate", {})
    checks = gate.get("checks", {})
    group_gates = g5.get("group_gates", {})
    return (
        formal_passed
        and gate.get("passed") is True
        and all(checks.get(key) is True for key in CAUSAL_G5_CHECKS)
        and all(
            isinstance(group_gates.get(group), dict)
            and group_gates[group].get("passed") is True
            for group in ("body", "left_hand", "right_hand")
        )
    )


def audit_completion(
    *,
    g0: dict[str, Any],
    g1: dict[str, Any],
    g2: dict[str, Any],
    g3: dict[str, Any],
    formal: dict[str, Any],
    g4: dict[str, Any],
    g5: dict[str, Any],
    g6: dict[str, Any],
    g7: dict[str, Any],
    prerequisites: dict[str, bool],
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Evaluate every final gate without promoting proxy or historical evidence."""
    formal_passed = (
        formal.get("contract_version") == "phase2r-formal-v1"
        and formal.get("passed") is True
        and _all_true(formal.get("checks"))
    )
    historical_g2 = _historical_g2_core_passed(g2)
    gates = {
        "G0": _g0_passed(g0),
        "G1": _g1_passed(g0, g1),
        # The redesigned final chain requires the historical volume/integrity
        # checks and the formal A1R/3D-target contract over the exact splits.
        "G2": historical_g2 and formal_passed,
        "G3": _g3_passed(g3),
        "G4": formal_passed and g4.get("passed") is True,
        "G5": _g5_passed(g5, formal_passed),
        "G6": g6.get("passed") is True
        and g6.get("checks", {}).get("exactly_three_seeds") is True,
        "G7": str(g7.get("decision", "")).startswith("GO"),
    }
    unmet_prerequisites = sorted(
        name for name, passed in prerequisites.items() if passed is not True
    )
    runtime_waits = sorted(
        name for name, value in runtime.items() if value not in {"COMPLETE", "PASS"}
    )
    failed_gates = [name for name, passed in gates.items() if not passed]
    full_go = not failed_gates
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": "FULL_GO" if full_go else "NO_GO",
        "full_go": full_go,
        "gates": gates,
        "failed_gates": failed_gates,
        "supporting_checks": {
            "historical_g2_volume_integrity": historical_g2,
            "formal_a1r_3d_target_contract": formal_passed,
            "g5_same_checkpoint_causal_checks_present_and_pass": all(
                g5.get("gate", {}).get("checks", {}).get(key) is True
                for key in CAUSAL_G5_CHECKS
            ),
        },
        "prerequisites": prerequisites,
        "unmet_prerequisites": unmet_prerequisites,
        "runtime": runtime,
        "runtime_waits": runtime_waits,
        "blocking_chain": [
            "licensed target intake and audit",
            "exact A1R materialization",
            "mesh-aligned G3/G4 candidate training",
            "causal U1 G5 calibration",
            "three-seed locked G6",
        ],
    }


def _validated_optional(path: Path | None, validator) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        validator(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _signer_map_available(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        payload = _load_json(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    mapping = payload.get("source_clip_to_signer", payload)
    return isinstance(mapping, dict) and bool(mapping) and all(
        str(key).strip() and str(value).strip() for key, value in mapping.items()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--g0")
    parser.add_argument("--g1")
    parser.add_argument("--g2")
    parser.add_argument("--g3")
    parser.add_argument("--formal-audit")
    parser.add_argument("--g4")
    parser.add_argument("--g5")
    parser.add_argument("--g6")
    parser.add_argument("--g7")
    parser.add_argument("--license-record", type=Path)
    parser.add_argument("--annotations-root", type=Path)
    parser.add_argument("--signer-map", type=Path)
    parser.add_argument("--target-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repository.resolve()
    defaults = {
        "g0": "outputs/phase2_gates/g0_a0/summary.json",
        "g1": "outputs/phase2_gates/g1_eval/method_ensemble/summary.json",
        "g2": "outputs/phase2_gates/g2/how2sign_2d_temporal_g2.json",
        "g3": (
            "outputs/phase2r/domain_aligned_v1_seed42/"
            "vertex_proxy_release_aligned_v2.json"
        ),
        "formal_audit": "outputs/phase2r/formal_preflight_current_proxy.json",
        "g4": (
            "outputs/phase2_gates/g4/"
            "how2sign_reprojection_v6_t5_formal_g4_decision.json"
        ),
        "g5": "outputs/phase2_gates/g5/how2sign_u1_v7_calibration_report.json",
        "g6": "outputs/phase2_gates/g6_reprojection_v6_t5/decision_seed42.json",
        "g7": "outputs/phase2_gates/g7/project_scope_author_1493_v1.json",
    }
    documents = {}
    for name, default in defaults.items():
        raw = getattr(args, name) or default
        path = Path(raw)
        documents[name] = _load_json(path if path.is_absolute() else root / path)

    annotations_available = bool(
        args.annotations_root
        and args.annotations_root.is_dir()
        and next(args.annotations_root.rglob("*.pkl"), None) is not None
    )
    report = audit_completion(
        g0=documents["g0"],
        g1=documents["g1"],
        g2=documents["g2"],
        g3=documents["g3"],
        formal=documents["formal_audit"],
        g4=documents["g4"],
        g5=documents["g5"],
        g6=documents["g6"],
        g7=documents["g7"],
        prerequisites={
            "signavatars_license_verified": _validated_optional(
                args.license_record, validate_license_acceptance
            ),
            "signavatars_annotations_available": annotations_available,
            "licensed_true_signer_map_available": _signer_map_available(
                args.signer_map
            ),
            "stratified_target_audit_passed": _validated_optional(
                args.target_audit, validate_target_audit
            ),
        },
        runtime={
            "exact_a1r_preflight": _status(
                root / "logs/phase2r/a1r_portable_preflight/status"
            ),
            "mesh_aligned_proxy": _status(
                root / "logs/phase2r/mesh_aligned_proxy_v2/status"
            ),
        },
    )
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["full_go"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
