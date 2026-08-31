"""Audit pseudo contact-event labels against an independently annotated gold tensor."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from dcg_sign4d.evaluation.contact_metrics import contact_event_metrics
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ: prediction,target,uncertain [T,E]")
    parser.add_argument("--edge-groups", required=True, help="JSON list with one group per edge")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--minimum-valid-support", type=int, required=True)
    parser.add_argument("--minimum-active-contact-f1", type=float, required=True)
    parser.add_argument("--minimum-segmental-f1", type=float, required=True)
    parser.add_argument("--maximum-onset-mae-sec", type=float, required=True)
    parser.add_argument("--maximum-release-mae-sec", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if args.minimum_valid_support < 1 or args.fps <= 0:
        raise ValueError("invalid support or FPS")
    for value in (args.minimum_active_contact_f1, args.minimum_segmental_f1):
        if not 0 <= value <= 1:
            raise ValueError("F1 thresholds must lie in [0,1]")
    if args.maximum_onset_mae_sec < 0 or args.maximum_release_mae_sec < 0:
        raise ValueError("timing thresholds cannot be negative")
    source = Path(args.input)
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != {"prediction", "target", "uncertain"}:
            raise ValueError("pseudo audit NPZ must contain prediction,target,uncertain")
        prediction = np.asarray(arrays["prediction"], dtype=np.int64)
        target = np.asarray(arrays["target"], dtype=np.int64)
        uncertain = np.asarray(arrays["uncertain"], dtype=np.bool_)
    if bool(((prediction < 0) | (prediction > 3)).any()):
        raise ValueError("prediction contains an invalid event state")
    if bool(((target < 0) | (target > 3)).any()):
        raise ValueError("target contains an invalid event state")
    edge_groups_path = Path(args.edge_groups)
    edge_groups = json.loads(edge_groups_path.read_text(encoding="utf-8"))
    if not isinstance(edge_groups, list) or len(edge_groups) != target.shape[1]:
        raise ValueError("edge group list must match E")
    allowed_groups = {"hand_hand", "hand_face", "hand_torso", "other"}
    if not all(group in allowed_groups for group in edge_groups):
        raise ValueError("unknown edge group")
    metrics = contact_event_metrics(prediction, target, uncertain, fps=args.fps)
    per_group = {}
    for group in sorted(set(edge_groups)):
        selected = np.asarray([value == group for value in edge_groups])
        per_group[group] = contact_event_metrics(
            prediction[:, selected],
            target[:, selected],
            uncertain[:, selected],
            fps=args.fps,
        )
    checks = {
        "valid_support": metrics["valid_support"] >= args.minimum_valid_support,
        "active_contact_f1": metrics["active_contact_f1"] >= args.minimum_active_contact_f1,
        "segmental_f1": metrics["segmental_f1_iou50"] >= args.minimum_segmental_f1,
        "onset_timing": math.isfinite(metrics["onset_timing_mae_sec"])
        and metrics["onset_timing_mae_sec"] <= args.maximum_onset_mae_sec,
        "release_timing": math.isfinite(metrics["release_timing_mae_sec"])
        and metrics["release_timing_mae_sec"] <= args.maximum_release_mae_sec,
    }
    gate_config = {
        "fps": args.fps,
        "minimum_valid_support": args.minimum_valid_support,
        "minimum_active_contact_f1": args.minimum_active_contact_f1,
        "minimum_segmental_f1": args.minimum_segmental_f1,
        "maximum_onset_mae_sec": args.maximum_onset_mae_sec,
        "maximum_release_mae_sec": args.maximum_release_mae_sec,
    }
    report = _clean(
        {
            "schema_version": "dcg_pseudo_contact_audit_v1",
            "scientific_scope": "PSEUDO_LABEL_QUALITY_COMPONENT_OF_G1",
            "development_only": args.development_only,
            "input_sha256": file_sha256(source),
            "edge_groups_sha256": file_sha256(edge_groups_path),
            "gate_config_sha256": canonical_hash(gate_config),
            "gate_config": gate_config,
            "metrics": metrics,
            "per_edge_group": per_group,
            "checks": checks,
            "pseudo_label_gate_pass": all(checks.values()),
        }
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable pseudo-label audit exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".pseudo_audit_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    (output / "pseudo_label_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    marker = "PSEUDO_LABEL_AUDIT_PASS" if all(checks.values()) else "PSEUDO_LABEL_AUDIT_FAILED"
    os.replace(incomplete, output / marker)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    if not all(checks.values()):
        raise RuntimeError("pseudo contact labels failed the preregistered gate")


if __name__ == "__main__":
    main()
