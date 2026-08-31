"""Freeze and audit the target-free arm BA V4 result tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from phase2_refiner.provenance import sha256_file
from signal4d_external.arm_ba_v4_core import ARM_JOINTS


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed result: {path}")
    return payload


def _equal(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return np.array_equal(np.asarray(first), np.asarray(second))
    return first == second


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    baseline = args.baseline_root.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("method") != "SIGNAL4D_EXTERNAL_ARM_BA_V4":
        raise ValueError("Unexpected run manifest")
    if int(manifest.get("sgnify_target_reads_before_evaluation", -1)) != 0:
        raise ValueError("Materialization does not prove zero target reads")
    tree = hashlib.sha256()
    frames = 0
    changed_arm_frames = 0
    exact_hand_arrays = 0
    exact_nonarm_body_joints = 0
    exact_other_fields = 0
    expected = {row["clip_id"]: row for row in manifest["clips"]}
    nonarm = np.asarray([joint for joint in range(21) if joint not in set(ARM_JOINTS)])
    for clip_id in sorted(expected):
        result_dir = root / clip_id / "smplifyx" / "results"
        paths = sorted(result_dir.glob("*.pkl"))
        if len(paths) != int(expected[clip_id]["frames"]):
            raise ValueError(f"Result coverage mismatch: {clip_id}")
        for output_path in paths:
            baseline_path = baseline / clip_id / "smplifyx" / "results" / output_path.name
            output = _load(output_path)
            reference = _load(baseline_path)
            if set(output) != set(reference):
                raise ValueError(f"Result keys changed: {output_path}")
            for key in output:
                if key != "body_pose":
                    if not _equal(output[key], reference[key]):
                        raise ValueError(f"Non-body field changed: {output_path}:{key}")
                    exact_other_fields += 1
            for key in ("left_hand_pose", "right_hand_pose"):
                if not _equal(output[key], reference[key]):
                    raise ValueError(f"Hand changed: {output_path}:{key}")
                exact_hand_arrays += 1
            body = np.asarray(output["body_pose"]).reshape(21, 3)
            base_body = np.asarray(reference["body_pose"]).reshape(21, 3)
            if not np.array_equal(body[nonarm], base_body[nonarm]):
                raise ValueError(f"Non-arm body joint changed: {output_path}")
            exact_nonarm_body_joints += len(nonarm)
            if not np.array_equal(body[ARM_JOINTS], base_body[ARM_JOINTS]):
                changed_arm_frames += 1
            relative = output_path.relative_to(root).as_posix()
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(sha256_file(output_path).encode("ascii"))
            tree.update(b"\n")
            frames += 1
    checks = {
        "57_clips": len(expected) == 57,
        "1493_frames": frames == 1493,
        "all_target_free_clips_accepted": int(manifest["accepted_clips"]) == 57,
        "only_arm_body_joints_changed": True,
        "hands_exact_v1": True,
        "target_reads_zero": True,
    }
    return {
        "schema_version": "signal4d.external_arm_ba_v4_freeze.v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "output_root": str(root),
        "baseline_root": str(baseline),
        "run_manifest_sha256": sha256_file(manifest_path),
        "baseline_manifest_sha256": sha256_file(baseline / "run_manifest.json"),
        "result_tree_sha256": tree.hexdigest(),
        "clips": len(expected),
        "frames": frames,
        "changed_arm_frames": changed_arm_frames,
        "exact_hand_arrays": exact_hand_arrays,
        "exact_nonarm_body_joints": exact_nonarm_body_joints,
        "exact_other_fields": exact_other_fields,
        "sgnify_target_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
