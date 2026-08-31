from __future__ import annotations

from pathlib import Path

import numpy as np

from signpccx.data.manifest import read_jsonl
from signpccx.io import atomic_write_json


def huber_location(values: np.ndarray, delta: float = 1.5, iterations: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(values.shape)
    estimate = np.median(values, axis=0)
    scale = np.median(np.abs(values - estimate), axis=0) * 1.4826 + 1e-6
    for _ in range(iterations):
        residual = (values - estimate) / scale
        weight = np.minimum(1.0, delta / (np.abs(residual) + 1e-8))
        estimate = (weight * values).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-8)
    return estimate.astype(np.float32)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))


def pose_diversity_feature(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[0] <= 21 or joints.shape[1] != 3:
        raise ValueError(joints.shape)
    pelvis = joints[0]
    left_shoulder, right_shoulder = joints[16], joints[17]
    left_elbow, right_elbow = joints[18], joints[19]
    left_wrist, right_wrist = joints[20], joints[21]
    shoulder_width = max(float(np.linalg.norm(left_shoulder - right_shoulder)), 1e-4)
    left_angle = _angle(left_shoulder - left_elbow, left_wrist - left_elbow)
    right_angle = _angle(right_shoulder - right_elbow, right_wrist - right_elbow)
    relative = np.concatenate(((left_wrist - pelvis)[:2], (right_wrist - pelvis)[:2])) / shoulder_width
    hand_distance = np.linalg.norm(left_wrist - right_wrist) / shoulder_width
    body_height_proxy = np.linalg.norm((left_shoulder + right_shoulder) * 0.5 - pelvis) / shoulder_width
    return np.asarray([shoulder_width, left_angle, right_angle, *relative, hand_distance, body_height_proxy], dtype=np.float32)


def farthest_point_indices(features: np.ndarray, count: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not len(features):
        raise ValueError(features.shape)
    count = min(int(count), len(features))
    scale = features.std(axis=0)
    normalized = (features - features.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    center_distance = np.square(normalized).sum(axis=1)
    selected = [int(np.argmax(center_distance))]
    nearest = np.square(normalized - normalized[selected[0]]).sum(axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.square(normalized - normalized[index]).sum(axis=1))
    return np.asarray(selected, dtype=np.int64)


def calibrate_shared_beta(
    frame_cache: Path,
    manifest_root: Path,
    output_npz: Path,
    calibration_frames: int = 20,
    huber_delta: float = 1.5,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        for record in read_jsonl(manifest_path):
            cache = frame_cache / "clips" / record.sign / f"{record.source_frame_id:06d}.npz"
            with np.load(cache, allow_pickle=False) as archive:
                beta = np.asarray(archive["smplx_beta"], dtype=np.float32)
                joints = np.asarray(archive["smplx_joints_parametric"], dtype=np.float32)
                confidence = np.asarray(archive["anchor_uv_confidence"], dtype=np.float32)
            if beta.shape != (10,) or not np.isfinite(beta).all() or not np.isfinite(joints).all():
                raise ValueError(f"invalid identity observation: {cache}")
            candidates.append({
                "sign": record.sign,
                "frame_id": record.source_frame_id,
                "beta": beta,
                "feature": pose_diversity_feature(joints),
                "confidence": float(np.clip(confidence, 0.0, 1.0).mean()),
            })
    if not candidates:
        raise RuntimeError("No identity candidates")
    # Exclude only the lowest-confidence tail; diversity selection handles pose redundancy.
    confidence = np.asarray([item["confidence"] for item in candidates])
    threshold = float(np.quantile(confidence, 0.25))
    eligible = [item for item in candidates if float(item["confidence"]) >= threshold]
    selected_indices = farthest_point_indices(np.stack([item["feature"] for item in eligible]), calibration_frames)
    selected = [eligible[index] for index in selected_indices]
    beta_values = np.stack([item["beta"] for item in selected])
    shared_beta = huber_location(beta_values, delta=huber_delta)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        beta=shared_beta,
        calibration_betas=beta_values,
        calibration_features=np.stack([item["feature"] for item in selected]),
        calibration_confidence=np.asarray([item["confidence"] for item in selected], dtype=np.float32),
        calibration_frame_ids=np.asarray([item["frame_id"] for item in selected], dtype=np.int64),
    )
    report = {
        "schema_version": "signpccx.identity.v1",
        "scope": "signer",
        "candidate_frames": len(candidates),
        "eligible_frames": len(eligible),
        "calibration_frames": len(selected),
        "selection": "confidence_q25_then_farthest_point_pose_diversity",
        "estimator": "huber_location",
        "huber_delta": huber_delta,
        "beta": shared_beta.tolist(),
        "selected": [{"sign": item["sign"], "frame_id": item["frame_id"], "confidence": item["confidence"]} for item in selected],
    }
    atomic_write_json(output_npz.with_suffix(".json"), report)
    return report

