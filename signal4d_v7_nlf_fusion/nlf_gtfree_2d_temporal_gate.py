#!/usr/bin/env python3
"""Materialize preregistered, zero-training SIGNAL-4D V7 predictions.

This program never loads SGNify ground truth or author-evaluation assets.  It
selects between frozen V6 and a fixed NLF/V6 SO(3) midpoint using only image
2D observations, camera intrinsics, NLF uncertainty, and temporal motion.
"""

from __future__ import annotations

import argparse
import csv
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
    geodesic_blend,
    preserve_global_rotations,
)


MODEL_JOINTS = np.asarray(list(range(1, 22)) + list(range(25, 55)), dtype=np.int64)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_id(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"Frame name has no numeric suffix: {name}")
    return int(match.group(1))


def load_observation_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    with (root / "index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (str(record["clip_id"]), int(record["frame_id"]))
            if key in result:
                raise ValueError(f"Duplicate NLF observation: {key}")
            result[key] = record
    return result


def project_normalized(
    joints: np.ndarray, camera: np.ndarray, image_size: np.ndarray
) -> np.ndarray:
    """Project stored author-protocol joints with the initializer camera.

    Author-protocol tensors have already received the x-axis 180-degree
    convention transform. The cached initializer intrinsics operate on the
    pre-transform SMPL-X camera coordinates, so the self-inverse transform is
    applied once before perspective projection.
    """
    camera_joints = joints * CAMERA_X_180
    z = np.maximum(camera_joints[..., 2], 1e-5)
    x = camera_joints[..., 0] / z * camera[0, 0] + camera[0, 2]
    y = camera_joints[..., 1] / z * camera[1, 1] + camera[1, 2]
    diagonal = float(np.linalg.norm(image_size.astype(np.float64)))
    return np.stack((x, y), axis=-1).astype(np.float32) / max(diagonal, 1.0)


def observed_normalized(points: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    height, width = image_size
    pixels = np.empty_like(points, dtype=np.float32)
    pixels[..., 0] = (points[..., 0] + 1.0) * 0.5 * width
    pixels[..., 1] = (points[..., 1] + 1.0) * 0.5 * height
    return pixels / max(float(np.linalg.norm(image_size.astype(np.float64))), 1.0)


def weighted_error(
    difference: np.ndarray, weights: np.ndarray, minimum_valid: int
) -> float:
    valid = np.isfinite(difference).all(axis=-1) & np.isfinite(weights) & (weights > 0)
    if int(valid.sum()) < minimum_valid:
        return float("inf")
    distances = np.linalg.norm(difference[valid], axis=-1)
    return float(np.average(distances, weights=weights[valid]))


def viterbi_select(
    observed: np.ndarray,
    projected: np.ndarray,
    weights: np.ndarray,
    candidate_valid: np.ndarray,
    minimum_valid: int,
    temporal_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose V6/candidate states with a deterministic image-space objective."""
    frames = observed.shape[0]
    if projected.shape[:2] != (frames, 2):
        raise ValueError(f"Expected projected [T,2,J,2], got {projected.shape}")
    unary = np.full((frames, 2), np.inf, dtype=np.float64)
    transition = np.full((frames, 2, 2), np.inf, dtype=np.float64)
    for t in range(frames):
        for state in range(2):
            if state == 1 and not candidate_valid[t]:
                continue
            unary[t, state] = weighted_error(
                projected[t, state] - observed[t], weights[t], minimum_valid
            )
    transition[0] = 0.0
    for t in range(1, frames):
        pair_weights = np.minimum(weights[t - 1], weights[t])
        observed_motion = observed[t] - observed[t - 1]
        for previous in range(2):
            for state in range(2):
                if (previous == 1 and not candidate_valid[t - 1]) or (
                    state == 1 and not candidate_valid[t]
                ):
                    continue
                predicted_motion = projected[t, state] - projected[t - 1, previous]
                transition[t, previous, state] = weighted_error(
                    predicted_motion - observed_motion, pair_weights, minimum_valid
                )

    costs = np.full((frames, 2), np.inf, dtype=np.float64)
    back = np.zeros((frames, 2), dtype=np.int8)
    costs[0] = unary[0]
    if not np.isfinite(costs[0, 0]):
        costs[0, 0] = 0.0
    for t in range(1, frames):
        for state in range(2):
            candidates = costs[t - 1] + temporal_weight * transition[t, :, state]
            # np.argmin resolves exact ties toward state 0 (V6).
            previous = int(np.argmin(candidates))
            costs[t, state] = unary[t, state] + candidates[previous]
            back[t, state] = previous
        if not np.isfinite(costs[t]).any():
            costs[t, 0] = costs[t - 1, 0]
            back[t, 0] = 0

    states = np.zeros(frames, dtype=np.int8)
    states[-1] = int(np.argmin(costs[-1]))
    for t in range(frames - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states.astype(bool), unary, transition


def _load_camera(source: str) -> np.ndarray:
    with Path(source).open("rb") as handle:
        params = pickle.load(handle, encoding="latin1")
    camera = np.asarray(params.get("K"), dtype=np.float32)
    if camera.shape != (3, 3):
        raise ValueError(f"Missing 3x3 K in {source}")
    return camera


def _fused_clip(
    sign: str,
    frame_ids: np.ndarray,
    v6: dict[str, np.ndarray],
    observation_index: dict[tuple[str, int], dict[str, Any]],
    observation_root: Path,
    baseline_parameter_root: Path,
    model: smplx.SMPLX,
    parents: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations, betas, expressions = [], [], []
    translations, plausibility = [], []
    for index_in_clip, value in enumerate(frame_ids):
        frame_id = int(value)
        record = observation_index[(sign, frame_id)]
        observation = np.load(observation_root / str(record["output_relpath"]))
        nlf = Rotation.from_rotvec(observation["pose"].reshape(55, 3)).as_matrix()
        fused = geodesic_blend(v6["rotations"][index_in_clip], nlf, alpha)
        fused[22:55] = v6["rotations"][index_in_clip, 22:55]
        fused = preserve_global_rotations(
            v6["rotations"][index_in_clip], fused, parents, (20, 21)
        )
        rotations.append(fused)
        parameter_path = (
            baseline_parameter_root
            / sign
            / "smplifyx"
            / "results"
            / f"low_{frame_id}.pkl"
        )
        with parameter_path.open("rb") as handle:
            baseline = pickle.load(handle, encoding="latin1")
        betas.append(np.asarray(baseline["betas"], dtype=np.float32).reshape(10))
        expressions.append(
            np.asarray(baseline["expression"], dtype=np.float32).reshape(10)
        )
        # Stored V6 tensors already carry the author-protocol x-180 transform;
        # SMPL-X forward expects the pre-transform camera translation.
        translations.append(v6["translation"][index_in_clip] * CAMERA_X_180)
        uncertainty = np.asarray(observation["joint_uncertainties"])[16:22]
        plausibility.append(float(np.mean(uncertainty < 250.0)))

    pose = Rotation.from_matrix(np.stack(rotations).reshape(-1, 3, 3)).as_rotvec()
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
    vertices = body.vertices.detach().cpu().numpy() * CAMERA_X_180
    joints = body.joints[:, :55].detach().cpu().numpy() * CAMERA_X_180
    return (
        vertices.astype(np.float32),
        joints.astype(np.float32),
        np.stack(rotations).astype(np.float32),
        np.asarray(plausibility, dtype=np.float32),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["supervision"]["trained_parameters"] != 0:
        raise ValueError("This executable permits zero trained parameters only")
    arm_cache = np.asarray(config["arm_cache_joint_indices"], dtype=np.int64)
    arm_smplx = MODEL_JOINTS[arm_cache]
    alpha = float(config["alpha"])
    temporal_weight = float(config["temporal_weight"])
    minimum_valid = int(config["minimum_valid_arm_joints"])
    threshold = float(config["nlf_plausibility"]["uncertainty_threshold_mm"])
    if threshold != 250.0:
        raise ValueError("Locked v1 implementation requires the 250 mm NLF threshold")
    minimum_fraction = float(
        config["nlf_plausibility"]["minimum_fraction_below_threshold"]
    )
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=False)
    observation_index = load_observation_index(args.observation_root)
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
    selection_rows: list[dict[str, Any]] = []
    selected_total = 0
    frames_total = 0

    cache_paths = sorted((args.cache_root / "clips").glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No cache clips under {args.cache_root / 'clips'}")
    for cache_path in cache_paths:
        cache = load_cache_clip(cache_path)
        sign = cache.clip_id
        v6_path = args.v6_root / sign / "prediction.safetensors"
        v6 = load_file(v6_path)
        frame_ids = v6["frame_ids"].astype(np.int64)
        cache_ids = np.asarray([_frame_id(str(name)) for name in cache.frame_names])
        if not np.array_equal(frame_ids, cache_ids):
            raise ValueError(f"Frame mismatch for {sign}")
        missing = [(sign, int(frame)) for frame in frame_ids if (sign, int(frame)) not in observation_index]
        if missing:
            raise FileNotFoundError(f"Missing NLF observations, first={missing[0]}")

        candidate_vertices, candidate_joints, candidate_rotations, plausibility = _fused_clip(
            sign,
            frame_ids,
            v6,
            observation_index,
            args.observation_root,
            args.baseline_parameter_root,
            model,
            parents,
            alpha,
        )
        frames = len(frame_ids)
        observed = np.stack(
            [observed_normalized(cache.keypoints_2d[t, arm_cache], cache.image_size[t]) for t in range(frames)]
        )
        projected = np.empty((frames, 2, len(arm_cache), 2), dtype=np.float32)
        for t in range(frames):
            camera = _load_camera(str(cache.source_paths[t]))
            projected[t, 0] = project_normalized(
                v6["joints_3d"][t, arm_smplx], camera, cache.image_size[t]
            )
            projected[t, 1] = project_normalized(
                candidate_joints[t, arm_smplx], camera, cache.image_size[t]
            )
        weights = np.where(
            cache.keypoint_valid[:, arm_cache],
            cache.u0_reliability[:, arm_cache],
            0.0,
        ).astype(np.float32)
        candidate_valid = plausibility >= minimum_fraction
        selected, unary, transition = viterbi_select(
            observed,
            projected,
            weights,
            candidate_valid,
            minimum_valid,
            temporal_weight,
        )
        output = {key: value.copy() for key, value in v6.items()}
        output["vertices"][selected] = candidate_vertices[selected]
        output["joints_3d"][selected] = candidate_joints[selected]
        output["rotations"][selected] = candidate_rotations[selected]

        destination = args.output_root / "predictions" / sign
        destination.mkdir(parents=True)
        save_file(output, destination / "prediction.safetensors")
        metadata = {
            "schema_version": config["schema_version"],
            "method_name": "SIGNAL4D_V7_GTFree2DTemporalGate",
            "clip_id": sign,
            "frames": frames,
            "selected_frames": int(selected.sum()),
            "training_frames": 0,
            "sgnify_gt_loaded": False,
            "config_sha256": sha256_file(args.config),
            "v6_artifact_sha256": sha256_file(v6_path),
            "artifact_sha256": sha256_file(destination / "prediction.safetensors"),
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for t, frame_id in enumerate(frame_ids):
            previous = int(selected[t - 1]) if t else 0
            state = int(selected[t])
            selection_rows.append(
                {
                    "sign": sign,
                    "frame": int(frame_id),
                    "selected_candidate": bool(selected[t]),
                    "candidate_plausibility_fraction": float(plausibility[t]),
                    "v6_unary": float(unary[t, 0]),
                    "candidate_unary": float(unary[t, 1]),
                    "selected_transition": float(transition[t, previous, state]),
                }
            )
        selected_total += int(selected.sum())
        frames_total += frames
        print(f"[v7-clean] {sign}: {int(selected.sum())}/{frames}", flush=True)

    with (args.output_root / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    report = {
        "schema_version": config["schema_version"],
        "method_name": "SIGNAL4D_V7_GTFree2DTemporalGate",
        "frames": frames_total,
        "selected_frames": selected_total,
        "selection_fraction": selected_total / frames_total,
        "training_frames": 0,
        "sgnify_ground_truth_loaded": False,
        "runtime_seconds": time.time() - started,
        "material_passport": {
            "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
            "nlf_observations": {"path": str(args.observation_root.resolve()), "metadata_sha256": sha256_file(args.observation_root / "run_metadata.json")},
            "v6_predictions": str(args.v6_root.resolve()),
            "image_observation_cache": str(args.cache_root.resolve()),
            "baseline_parameters": str(args.baseline_parameter_root.resolve()),
            "smplx_model": {"path": str(args.model_path.resolve()), "sha256": sha256_file(args.model_path)},
        },
        "claim_boundary": "No SGNify GT was loaded, no parameter was trained, and no threshold or alpha was selected by author-protocol metrics.",
    }
    (args.output_root / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--observation-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--v6-root", required=True, type=Path)
    parser.add_argument("--baseline-parameter-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
