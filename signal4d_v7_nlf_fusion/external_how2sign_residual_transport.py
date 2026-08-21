#!/usr/bin/env python3
"""Transport a frozen How2Sign-trained SO(3) residual onto frozen V6.

The program consumes no SGNify target or author-evaluation artifact. The
external checkpoint was trained/selected on source-disjoint How2Sign data.
Its already-materialized Lane inference supplies a local rotation residual,
which is transported to V6 rather than replacing the V6 pose. V6 hands and
global wrist orientations are preserved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
from safetensors.numpy import load_file, save_file
from scipy.spatial.transform import Rotation

from phase2_refiner.data.cache_schema import load_cache_clip
from signal4d_v7_nlf_fusion.nlf_body_router import (
    CAMERA_X_180,
    preserve_global_rotations,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def frame_id(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"Frame name has no numeric suffix: {name}")
    return int(match.group(1))


def pose51(params: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(params["body_pose"], dtype=np.float32).reshape(21, 3),
            np.asarray(params["left_hand_pose"], dtype=np.float32).reshape(15, 3),
            np.asarray(params["right_hand_pose"], dtype=np.float32).reshape(15, 3),
        ),
        axis=0,
    )


def transport_local_residual(
    reference: np.ndarray,
    source_initial: np.ndarray,
    source_refined: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Left-transport a source local SO(3) residual onto a reference pose."""
    delta = source_refined @ np.swapaxes(source_initial, -1, -2)
    tangent = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec()
    step = Rotation.from_rotvec(alpha * tangent).as_matrix().reshape(delta.shape)
    return (step @ reference).astype(np.float32)


def audit_external_checkpoint(
    checkpoint_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != expected_sha256:
        raise ValueError(f"External checkpoint hash mismatch: {actual_hash}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError("Checkpoint has no resolved_config provenance")
    data = resolved.get("data", {})
    declared_train = str(data.get("train_glob", "")).lower()
    declared_val = str(data.get("val_glob", "")).lower()
    for split, value in (("train", declared_train), ("val", declared_val)):
        if "how2sign" not in value or "sgnify" in value or "smplx_gt" in value:
            raise ValueError(f"Non-external {split} path in checkpoint: {value}")
    for path in (train_manifest, val_manifest):
        payload = json.loads(path.read_text(encoding="utf-8"))
        clips = payload.get("clips", [])
        if (
            payload.get("dataset") != "How2Sign"
            or payload.get("sgnify_excluded") is not True
            or not clips
            or any("how2sign" not in str(value).lower() for value in clips)
        ):
            raise ValueError(f"Manifest is not How2Sign-only: {path}")
    return {
        "checkpoint_sha256": actual_hash,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_best_external_score": float(checkpoint.get("best", float("nan"))),
        "train_manifest_sha256": sha256_file(train_manifest),
        "val_manifest_sha256": sha256_file(val_manifest),
        "declared_train": str(data.get("train_glob")),
        "declared_val": str(data.get("val_glob")),
        "sgnify_training_or_calibration_frames": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    alpha = float(config["residual_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("residual_alpha must be in [0,1]")
    external_audit = audit_external_checkpoint(
        args.external_checkpoint,
        args.external_train_manifest,
        args.external_val_manifest,
        str(config["external_checkpoint_sha256"]),
    )
    source_run = json.loads(
        (args.external_prediction_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if source_run.get("checkpoint_sha256") != external_audit["checkpoint_sha256"]:
        raise ValueError("Lane inference was not produced by the locked external checkpoint")
    args.output_root.mkdir(parents=True, exist_ok=False)
    model_hash = sha256_file(args.model_path)
    model = smplx.SMPLX(
        str(args.model_path),
        gender="neutral",
        ext=args.model_path.suffix.lstrip("."),
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).eval()
    parents = model.parents[:55].detach().cpu().numpy().astype(np.int64)
    body_cache_indices = np.asarray(config["body_cache_joint_indices"], dtype=np.int64)
    body_smplx_indices = body_cache_indices + 1
    preserve_globals = tuple(int(v) for v in config["preserve_global_smplx_joint_indices"])
    frames_total = 0
    max_global_wrist_error = 0.0
    max_closed_rotation_error = 0.0
    clip_rows: list[dict[str, Any]] = []

    cache_paths = sorted((args.cache_root / "clips").glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No cache clips under {args.cache_root / 'clips'}")
    for cache_path in cache_paths:
        clip = load_cache_clip(cache_path)
        if clip.target_axis_angle is not None or clip.target_joint_positions is not None:
            raise ValueError(f"Inference cache unexpectedly contains targets: {cache_path}")
        sign = clip.clip_id
        v6_path = args.v6_root / sign / "prediction.safetensors"
        v6 = load_file(v6_path)
        frame_ids = v6["frame_ids"].astype(np.int64)
        cache_ids = np.asarray([frame_id(str(value)) for value in clip.frame_names])
        if not np.array_equal(frame_ids, cache_ids):
            raise ValueError(f"Frame mismatch for {sign}")

        candidate_rotations: list[np.ndarray] = []
        betas, expressions, translations = [], [], []
        for t, name in enumerate(clip.frame_names.astype(str)):
            source_initial = Rotation.from_rotvec(clip.init_axis_angle[t]).as_matrix()
            refined_path = (
                args.external_prediction_root / sign / "smplifyx" / "results" / f"{name}.pkl"
            )
            with refined_path.open("rb") as handle:
                refined_params = pickle.load(handle, encoding="latin1")
            source_refined = Rotation.from_rotvec(pose51(refined_params)).as_matrix()
            fused = v6["rotations"][t].copy()
            fused[body_smplx_indices] = transport_local_residual(
                v6["rotations"][t, body_smplx_indices],
                source_initial[body_cache_indices],
                source_refined[body_cache_indices],
                alpha,
            )
            fused[22:55] = v6["rotations"][t, 22:55]
            fused = preserve_global_rotations(
                v6["rotations"][t], fused, parents, preserve_globals
            )
            candidate_rotations.append(fused)

            baseline_path = (
                args.baseline_parameter_root
                / sign
                / "smplifyx"
                / "results"
                / f"low_{int(frame_ids[t])}.pkl"
            )
            with baseline_path.open("rb") as handle:
                baseline = pickle.load(handle, encoding="latin1")
            betas.append(np.asarray(baseline["betas"], dtype=np.float32).reshape(10))
            expressions.append(np.asarray(baseline["expression"], dtype=np.float32).reshape(10))
            translations.append(v6["translation"][t] * CAMERA_X_180)

        rotations = np.stack(candidate_rotations).astype(np.float32)
        pose = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec()
        pose = torch.from_numpy(pose.reshape(len(frame_ids), 55, 3).astype(np.float32))
        with torch.inference_mode():
            body = model(
                global_orient=pose[:, 0],
                body_pose=pose[:, 1:22].flatten(1),
                jaw_pose=pose[:, 22],
                leye_pose=pose[:, 23],
                reye_pose=pose[:, 24],
                left_hand_pose=pose[:, 25:40].flatten(1),
                right_hand_pose=pose[:, 40:55].flatten(1),
                betas=torch.from_numpy(np.stack(betas)),
                expression=torch.from_numpy(np.stack(expressions)),
                transl=torch.from_numpy(np.stack(translations)),
                return_verts=True,
            )
        vertices = (body.vertices.detach().cpu().numpy() * CAMERA_X_180).astype(np.float32)
        joints = (body.joints[:, :55].detach().cpu().numpy() * CAMERA_X_180).astype(np.float32)
        output = {key: value.copy() for key, value in v6.items()}
        output["rotations"] = rotations
        output["vertices"] = vertices
        output["joints_3d"] = joints

        closed = np.ones(55, dtype=bool)
        closed[body_smplx_indices] = False
        closed[list(preserve_globals)] = False
        closed_error = float(np.max(np.abs(rotations[:, closed] - v6["rotations"][:, closed])))
        max_closed_rotation_error = max(max_closed_rotation_error, closed_error)
        reference_global = np.stack(
            [
                _global_rotations(v6["rotations"][t], parents)
                for t in range(len(frame_ids))
            ]
        )
        candidate_global = np.stack(
            [_global_rotations(rotations[t], parents) for t in range(len(frame_ids))]
        )
        wrist_error = float(
            np.max(np.abs(candidate_global[:, preserve_globals] - reference_global[:, preserve_globals]))
        )
        max_global_wrist_error = max(max_global_wrist_error, wrist_error)
        if closed_error > 1e-6 or wrist_error > 1e-5:
            raise AssertionError(
                f"Preservation contract failed for {sign}: closed={closed_error}, wrist={wrist_error}"
            )

        destination = args.output_root / "predictions" / sign
        destination.mkdir(parents=True)
        save_file(output, destination / "prediction.safetensors")
        metadata = {
            "schema_version": config["schema_version"],
            "method_name": config["method_name"],
            "clip_id": sign,
            "frames": int(len(frame_ids)),
            "frame_ids": frame_ids.tolist(),
            "training_corpus": "How2Sign only",
            "sgnify_training_frames": 0,
            "sgnify_calibration_frames": 0,
            "sgnify_gt_loaded": False,
            "coordinate_convention": "opencv_x_right_y_down_z_forward",
            "length_unit": "meter",
            "smplx_model_sha256": model_hash,
            "config_sha256": sha256_file(args.config),
            "external_checkpoint_sha256": external_audit["checkpoint_sha256"],
            "v6_artifact_sha256": sha256_file(v6_path),
            "artifact_sha256": sha256_file(destination / "prediction.safetensors"),
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frames_total += len(frame_ids)
        clip_rows.append(
            {
                "clip_id": sign,
                "frames": int(len(frame_ids)),
                "max_closed_rotation_error": closed_error,
                "max_global_wrist_error": wrist_error,
            }
        )
        print(f"[v7-ext] {sign}: {len(frame_ids)} frames", flush=True)

    report = {
        "schema_version": config["schema_version"],
        "method_name": config["method_name"],
        "frames": frames_total,
        "clips": len(clip_rows),
        "training_corpus": "How2Sign only",
        "sgnify_training_or_calibration_frames": 0,
        "sgnify_gt_loaded": False,
        "max_closed_rotation_error": max_closed_rotation_error,
        "max_global_wrist_error": max_global_wrist_error,
        "external_training_audit": external_audit,
        "runtime_seconds": time.time() - started,
        "material_passport": {
            "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
            "external_checkpoint": str(args.external_checkpoint.resolve()),
            "external_prediction_run_manifest": {
                "path": str((args.external_prediction_root / "run_manifest.json").resolve()),
                "sha256": sha256_file(args.external_prediction_root / "run_manifest.json"),
            },
            "inference_cache": {"path": str(args.cache_root.resolve()), "contains_targets": False},
            "v6_predictions": str(args.v6_root.resolve()),
            "smplx_model": {"path": str(args.model_path.resolve()), "sha256": model_hash},
        },
        "claim_boundary": (
            "Checkpoint fitting and checkpoint selection use How2Sign only. SGNify supplies "
            "image observations at inference, never targets. Historical SGNify benchmark "
            "results were inspected before this integration and must be disclosed."
        ),
    }
    (args.output_root / "clip_audit.json").write_text(
        json.dumps(clip_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _global_rotations(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    output = np.empty_like(local)
    for index, parent in enumerate(parents):
        output[index] = local[index] if int(parent) < 0 else output[int(parent)] @ local[index]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--external-prediction-root", required=True, type=Path)
    parser.add_argument("--external-checkpoint", required=True, type=Path)
    parser.add_argument("--external-train-manifest", required=True, type=Path)
    parser.add_argument("--external-val-manifest", required=True, type=Path)
    parser.add_argument("--v6-root", required=True, type=Path)
    parser.add_argument("--baseline-parameter-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
