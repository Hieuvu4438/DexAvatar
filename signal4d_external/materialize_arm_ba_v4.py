"""Materialize frozen external-only arm BA V4 on target-free inference caches."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.refine_how2sign_targets import create_smplx_model
from phase2_refiner.provenance import run_provenance, sha256_file
from signal4d_external.arm_ba_v4_core import (
    ARM_JOINTS,
    fit_arm_batch,
    source_intrinsics,
)
from signal4d_external.nlf_v2_core import geodesic_blend


def _baseline_pose(
    clip: Any, baseline_root: Path
) -> tuple[np.ndarray, list[dict[str, Any]], list[Path]]:
    poses = []
    payloads = []
    paths = []
    for frame_name in clip.frame_names:
        path = baseline_root / clip.clip_id / "smplifyx" / "results" / f"{frame_name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        body = np.asarray(payload["body_pose"], dtype=np.float32).reshape(21, 3)
        left = np.asarray(payload["left_hand_pose"], dtype=np.float32).reshape(15, 3)
        right = np.asarray(payload["right_hand_pose"], dtype=np.float32).reshape(15, 3)
        poses.append(np.concatenate((body, left, right), axis=0))
        payloads.append(payload)
        paths.append(path)
    return np.stack(poses), payloads, paths


def _materialize_clip(
    model: Any,
    clip: Any,
    baseline_root: Path,
    output_root: Path,
    calibration: dict[str, Any],
    device: torch.device,
    minimum_reprojection_gain: float,
) -> dict[str, Any]:
    initial, payloads, baseline_paths = _baseline_pose(clip, baseline_root)
    working = replace(clip, init_axis_angle=initial.astype(np.float32))
    candidate, reports = fit_arm_batch(
        model,
        [working],
        observed=working.keypoints_2d[None],
        confidence=working.calibrated_confidence[None],
        valid=working.keypoint_valid[None],
        device=device,
        projection="intrinsics",
        projection_aux=(source_intrinsics(working)[None], working.image_size[None]),
        iterations=int(calibration["parameters"]["iterations"]),
        learning_rate=float(calibration["parameters"]["learning_rate"]),
        max_degrees=float(calibration["parameters"]["max_degrees"]),
    )
    blend = float(calibration["selected"]["blend"])
    baseline_matrix = Rotation.from_rotvec(initial.reshape(-1, 3)).as_matrix().reshape(
        -1, 51, 3, 3
    )
    candidate_matrix = Rotation.from_rotvec(candidate.reshape(-1, 3)).as_matrix().reshape(
        -1, 51, 3, 3
    )
    blended = geodesic_blend(baseline_matrix, candidate_matrix, blend)
    output_pose = Rotation.from_matrix(blended.reshape(-1, 3, 3)).as_rotvec().reshape(
        -1, 51, 3
    ).astype(np.float32)
    reprojection_gain = float(reports[0]["relative_gain"])
    accepted = bool(
        np.isfinite(reprojection_gain) and reprojection_gain >= minimum_reprojection_gain
    )
    if not accepted:
        output_pose = initial.copy()
    # Only arm body joints may differ.  Hands and non-arm body joints are copied
    # from the exact V1 payload below, independent of numerical conversions.
    result_dir = output_root / clip.clip_id / "smplifyx" / "results"
    diagnostics_dir = output_root / clip.clip_id / "external_arm_ba_v4"
    result_dir.mkdir(parents=True)
    diagnostics_dir.mkdir(parents=True)
    for index, (frame_name, payload) in enumerate(zip(clip.frame_names, payloads, strict=True)):
        refined = dict(payload)
        body = np.asarray(payload["body_pose"], dtype=np.float32).reshape(21, 3).copy()
        if accepted:
            body[ARM_JOINTS] = output_pose[index, ARM_JOINTS]
        refined["body_pose"] = body.reshape(1, 63)
        destination = result_dir / f"{frame_name}.pkl"
        with destination.open("xb") as handle:
            pickle.dump(refined, handle, protocol=2)
    delta = candidate_matrix[:, ARM_JOINTS] @ np.swapaxes(
        baseline_matrix[:, ARM_JOINTS], -1, -2
    )
    movement = np.rad2deg(
        np.linalg.norm(
            Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
        )
    ).reshape(len(initial), len(ARM_JOINTS))
    diagnostics_path = diagnostics_dir / "sequence.npz"
    np.savez_compressed(
        diagnostics_path,
        frame_names=clip.frame_names,
        arm_movement_degrees=movement,
        accepted=np.asarray(accepted),
        blend=np.asarray(blend),
        reprojection_gain=np.asarray(reprojection_gain),
    )
    return {
        "clip_id": clip.clip_id,
        "frames": len(clip.frame_names),
        "accepted": accepted,
        "reprojection_gain": reprojection_gain,
        "initial_reprojection": float(reports[0]["initial_reprojection"]),
        "final_reprojection": float(reports[0]["final_reprojection"]),
        "mean_arm_movement_degrees": float(movement.mean()) if accepted else 0.0,
        "max_arm_movement_degrees": float(movement.max()) if accepted else 0.0,
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "baseline_results": [str(path.resolve()) for path in baseline_paths],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("decision") != "PASS":
        raise ValueError("External calibration did not pass")
    model_file = args.model_folder.resolve() / "smplx" / "SMPLX_NEUTRAL.npz"
    if sha256_file(model_file) != calibration.get("model_sha256"):
        raise ValueError("SMPL-X model does not match external calibration")
    cache_manifest = args.cache_root.resolve() / "manifest.json"
    cache_payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if any(bool(row.get("has_target")) for row in cache_payload.get("clips", [])):
        raise ValueError("Inference manifest contains target fields")
    cache_paths = sorted((args.cache_root.resolve() / "clips").glob("*.npz"))
    if len(cache_paths) != 57:
        raise ValueError(f"Expected 57 inference clips, got {len(cache_paths)}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    model = create_smplx_model(args.model_folder.resolve(), device)
    model.requires_grad_(False)
    summaries = []
    for index, cache_path in enumerate(cache_paths, start=1):
        clip = load_cache_clip(cache_path)
        if clip.target_axis_angle is not None or clip.target_joint_positions is not None:
            raise ValueError(f"Target fields present: {cache_path}")
        metadata = json.loads(clip.metadata_json)
        if int(metadata.get("sgnify_target_reads", 0)) != 0:
            raise ValueError(f"Cache reports target reads: {cache_path}")
        summary = _materialize_clip(
            model,
            clip,
            args.baseline_root.resolve(),
            args.output.resolve(),
            calibration,
            device,
            args.minimum_reprojection_gain,
        )
        summary["cache"] = str(cache_path.resolve())
        summary["cache_sha256"] = sha256_file(cache_path)
        summaries.append(summary)
        print(
            f"[arm-ba-v4-target] {index}/{len(cache_paths)} {clip.clip_id} "
            f"accepted={summary['accepted']} gain={summary['reprojection_gain']:.4f}",
            flush=True,
        )
    frame_count = sum(row["frames"] for row in summaries)
    if frame_count != 1493:
        raise ValueError(f"Expected 1493 frames, got {frame_count}")
    manifest = {
        "schema_version": "signal4d.external_arm_ba_v4_target.v1",
        "method": "SIGNAL4D_EXTERNAL_ARM_BA_V4",
        "frames": frame_count,
        "clips": summaries,
        "accepted_clips": int(sum(row["accepted"] for row in summaries)),
        "minimum_reprojection_gain": args.minimum_reprojection_gain,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "baseline_root": str(args.baseline_root.resolve()),
        "baseline_manifest_sha256": sha256_file(
            args.baseline_root.resolve() / "run_manifest.json"
        ),
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "model_sha256": sha256_file(model_file),
        "sgnify_target_reads_before_evaluation": 0,
        "provenance": run_provenance(args.calibration, 42),
    }
    manifest_path = args.output.resolve() / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--model-folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-reprojection-gain", type=float, default=0.005)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
