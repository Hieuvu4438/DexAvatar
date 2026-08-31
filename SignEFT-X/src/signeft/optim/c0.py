from __future__ import annotations

import json
from pathlib import Path

from signeft.data.manifest import read_manifest
from signeft.gating.rollback import exact_rollback
from signeft.io_utils import atomic_write_json, sha256_file


def run_c0_exact_baseline(manifest: Path, run_root: Path) -> dict[str, object]:
    records = read_manifest(manifest)
    decisions = []
    for record in records:
        baseline_obj = Path(record.a3f_obj_path)
        baseline_state = Path(record.a3f_state_path)
        if sha256_file(baseline_obj) != record.sha256_a3f_obj:
            raise RuntimeError(f"baseline OBJ changed: {record.record_id}")
        if sha256_file(baseline_state) != record.sha256_a3f_state:
            raise RuntimeError(f"baseline state changed: {record.record_id}")
        output_obj = run_root / "frames" / record.sign_id / f"{record.source_frame_id:06d}.obj"
        output_state = run_root / "frames" / record.sign_id / f"{record.source_frame_id:06d}.npz"
        decision_path = run_root / "decisions" / record.sign_id / f"{record.source_frame_id:06d}.json"
        if output_obj.is_file() and output_state.is_file() and decision_path.is_file():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if decision["output_hashes"]["obj"] != sha256_file(output_obj):
                raise RuntimeError(f"resume OBJ hash mismatch: {record.record_id}")
            if decision["output_hashes"]["state"] != sha256_file(output_state):
                raise RuntimeError(f"resume state hash mismatch: {record.record_id}")
            decisions.append(decision)
            continue
        hashes = exact_rollback(baseline_obj, baseline_state, output_obj, output_state)
        decision = {
            "schema_version": "signeft.decision.v1",
            "record_id": record.record_id,
            "candidate_id": "C0_A3F",
            "accepted": False,
            "winning_families": [],
            "losing_families": [],
            "energy_delta": {},
            "noise_sigma": {},
            "trust_max_deg": {},
            "lhand_centered_drift_mm": 0.0,
            "rhand_centered_drift_mm": 0.0,
            "off_target_drift_mm": {"face": 0.0, "lower_body": 0.0},
            "fallback": "exact_a3f",
            "reason": "C0_CONTROL_EXACT_BASELINE",
            "input_hashes": {
                "rgb": record.sha256_rgb,
                "obj": hashes["baseline_obj"],
                "state": hashes["baseline_state"],
            },
            "output_hashes": {
                "obj": hashes["output_obj"],
                "state": hashes["output_state"],
            },
            "objective_uses_ground_truth": False,
            "objective_uses_temporal_pose": False,
        }
        atomic_write_json(decision_path, decision)
        decisions.append(decision)
    summary = {
        "schema_version": "signeft.refinement-summary.v1",
        "status": "ok",
        "method": "C0_A3F_EXACT_CONTROL",
        "frames": len(records),
        "accepted": 0,
        "fallback": len(records),
        "manifest_sha256": sha256_file(manifest),
        "all_obj_exact": all(
            item["input_hashes"]["obj"] == item["output_hashes"]["obj"]
            for item in decisions
        ),
        "all_state_exact": all(
            item["input_hashes"]["state"] == item["output_hashes"]["state"]
            for item in decisions
        ),
    }
    atomic_write_json(run_root / "refinement_summary.json", summary)
    return summary

