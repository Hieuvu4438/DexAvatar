"""Machine-readable G0 evaluator perturbation and source-identity audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dcg_sign4d.evaluation.hand_metrics import HandPlacementMetrics
from dcg_sign4d.utils.hashing import file_sha256


def _fixture() -> tuple[HandPlacementMetrics, np.ndarray]:
    regressor = np.zeros((22, 8), dtype=np.float64)
    regressor[1, 0] = 1
    regressor[2, 1] = 1
    regressor[20, 2:4] = 0.5
    regressor[21, 4:6] = 0.5
    target = np.zeros((8, 3), dtype=np.float64)
    target[0, 0], target[1, 0] = -0.1, 0.1
    target[2:4, 0] = [-0.6, -0.5]
    target[4:6, 0] = [0.5, 0.6]
    target[6:, 1] = [0.2, 0.3]
    evaluator = HandPlacementMetrics(
        regressor,
        np.array([2, 3]),
        np.array([4, 5]),
        np.arange(8),
    )
    return evaluator, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-evaluator", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--coordinate-policy", required=True, choices=("author_frozen", "unfrozen"))
    parser.add_argument(
        "--signer-policy", required=True, choices=("signer_ids_available", "unavailable")
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluator, target = _fixture()
    translated = evaluator.evaluate_frame(target + np.array([3.0, -2.0, 8.0]), target)
    placement_source = target.copy()
    placement_source[[2, 3], 1] += 0.05
    placement = evaluator.evaluate_frame(placement_source, target)
    articulation_source = target.copy()
    articulation_source[2, 1] += 0.05
    articulation = evaluator.evaluate_frame(articulation_source, target)
    engineering_checks = {
        "global_translation_invariance": max(abs(value) for value in translated.values()) < 1e-9,
        "primary_detects_50mm_hand_placement": abs(
            placement["root_aligned_left_hand_pve_mm"] - 50.0
        )
        < 1e-9,
        "wrist_metric_removes_rigid_placement": placement["wrist_aligned_left_hand_pve_mm"] < 1e-9,
        "legacy_region_metric_removes_rigid_placement": placement[
            "legacy_region_tr_left_hand_pve_mm"
        ]
        < 1e-9,
        "wrist_metric_detects_articulation": articulation["wrist_aligned_left_hand_pve_mm"] > 0,
    }
    scientific_checks = {
        "coordinate_policy_frozen": args.coordinate_policy == "author_frozen",
        "signer_policy_available": args.signer_policy == "signer_ids_available",
    }
    report = {
        "schema_version": "dcg_g0_evaluator_audit_v1",
        "gate": "G0",
        "engineering_status": "PASS" if all(engineering_checks.values()) else "FAIL",
        "scientific_status": (
            "PASS"
            if all(engineering_checks.values()) and all(scientific_checks.values())
            else "BLOCKED"
        ),
        "author_evaluator_sha256": file_sha256(args.author_evaluator),
        "manifest_sha256": file_sha256(args.manifest),
        "engineering_checks": engineering_checks,
        "scientific_checks": scientific_checks,
        "perturbation_metrics_mm": {
            "placement_root_aligned": placement["root_aligned_left_hand_pve_mm"],
            "placement_wrist_aligned": placement["wrist_aligned_left_hand_pve_mm"],
            "placement_legacy_region": placement["legacy_region_tr_left_hand_pve_mm"],
            "articulation_wrist_aligned": articulation["wrist_aligned_left_hand_pve_mm"],
        },
        "interpretation": (
            "The attached-author region-centering endpoint removes rigid hand placement; "
            "root/pelvis alignment is the DCG primary endpoint."
        ),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable G0 audit exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".g0_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    (output / "G0_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    marker = "G0_PASS" if report["scientific_status"] == "PASS" else "G0_BLOCKED"
    os.replace(incomplete, output / marker)
    print(json.dumps(report, sort_keys=True, indent=2))
    if report["engineering_status"] != "PASS":
        raise RuntimeError("G0 evaluator engineering perturbation audit failed")


if __name__ == "__main__":
    main()
