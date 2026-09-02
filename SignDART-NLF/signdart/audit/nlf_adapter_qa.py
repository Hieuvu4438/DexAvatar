from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signdart.io.h1_state import read_manifest, sha256_file


ARM_IDS = np.asarray([13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int64)
REQUIRED = (
    "joints3d", "joints3d_nonparam", "joints2d", "joints2d_nonparam",
    "joint_uncertainties",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nlf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    index = {}
    with (args.nlf_root / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index[(str(row["clip_id"]), int(row["frame_id"]))] = row
    missing = []
    finite_frames = 0
    positive_uncertainty_frames = 0
    disagreement_3d = []
    disagreement_2d = []
    inside = []
    for record in records:
        key = (record["sign_id"], int(record["source_frame_id"]))
        if key not in index:
            missing.append(record["record_id"])
            continue
        with np.load(args.nlf_root / index[key]["output_relpath"]) as archive:
            arrays = {name: np.asarray(archive[name]) for name in REQUIRED}
        expected = {
            "joints3d": (55, 3), "joints3d_nonparam": (55, 3),
            "joints2d": (55, 2), "joints2d_nonparam": (55, 2),
            "joint_uncertainties": (55,),
        }
        shapes_ok = all(arrays[name].shape == expected[name] for name in REQUIRED)
        finite = shapes_ok and all(np.isfinite(value).all() for value in arrays.values())
        finite_frames += int(finite)
        positive_uncertainty_frames += int(
            finite and np.all(arrays["joint_uncertainties"] > 0)
        )
        if not finite:
            continue
        disagreement_3d.extend(np.linalg.norm(
            arrays["joints3d"][ARM_IDS] - arrays["joints3d_nonparam"][ARM_IDS], axis=1
        ).tolist())
        disagreement_2d.extend(np.linalg.norm(
            arrays["joints2d"][ARM_IDS] - arrays["joints2d_nonparam"][ARM_IDS], axis=1
        ).tolist())
        xy = arrays["joints2d_nonparam"][ARM_IDS]
        inside.extend((
            (xy[:, 0] >= 0) & (xy[:, 0] < int(record["width"]))
            & (xy[:, 1] >= 0) & (xy[:, 1] < int(record["height"]))
        ).tolist())
    frames = len(records)
    metrics = {
        "coverage": (frames - len(missing)) / frames,
        "finite_frame_fraction": finite_frames / frames,
        "positive_uncertainty_frame_fraction": positive_uncertainty_frames / frames,
        "median_arm_param_nonparam_3d_mm": float(np.median(disagreement_3d)),
        "median_arm_param_nonparam_2d_px": float(np.median(disagreement_2d)),
        "arm_joint_inside_image_fraction": float(np.mean(inside)),
    }
    limits = {
        "coverage": 1.0,
        "finite_frame_fraction": 1.0,
        "positive_uncertainty_frame_fraction": 1.0,
        "median_arm_param_nonparam_3d_mm_max": 25.0,
        "median_arm_param_nonparam_2d_px_max": 10.0,
        "arm_joint_inside_image_fraction_min": 0.9,
    }
    passed = bool(
        metrics["coverage"] == 1.0
        and metrics["finite_frame_fraction"] == 1.0
        and metrics["positive_uncertainty_frame_fraction"] == 1.0
        and metrics["median_arm_param_nonparam_3d_mm"] <= 25.0
        and metrics["median_arm_param_nonparam_2d_px"] <= 10.0
        and metrics["arm_joint_inside_image_fraction"] >= 0.9
    )
    report = {
        "schema_version": "signdart.nlf_adapter_qa.v1",
        "status": "pass" if passed else "fail",
        "frames": frames,
        "metrics": metrics,
        "limits": limits,
        "missing": missing,
        "manifest_sha256": sha256_file(args.manifest),
        "nlf_metadata_sha256": sha256_file(args.nlf_root / "run_metadata.json"),
        "uses_gt": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
