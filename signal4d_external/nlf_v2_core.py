"""Shared, target-free feature and SO(3) utilities for external-only V2.

The functions in this module deliberately know nothing about SGNify ground
truth.  Training callers may compute labels from external cache targets, while
inference callers can construct the exact same features from observations and
the frozen initializer alone.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


BODY_COUNT = 21
LEFT_HAND = slice(21, 36)
RIGHT_HAND = slice(36, 51)
NLF_BODY = slice(1, 22)
NLF_LEFT_HAND = slice(25, 40)
NLF_RIGHT_HAND = slice(40, 55)
GLOBAL_WRISTS = (20, 21)

# These are SMPL-X body-pose indices (root is stored separately).  The torso
# set supplies stable scale/context; the arm set carries the signing signal.
TORSO = np.asarray((2, 5, 8, 11, 12, 14), dtype=np.int64)
ARMS = np.asarray((15, 16, 17, 18, 19, 20), dtype=np.int64)
UPPER_BODY = np.unique(np.concatenate((TORSO, ARMS)))

FEATURE_COLUMNS = (
    "nlf_unc_torso",
    "nlf_unc_arms",
    "nlf_fit2d_torso",
    "nlf_fit2d_arms",
    "nlf_tals2d_torso",
    "nlf_tals2d_arms",
    "initializer_reprojection_torso",
    "initializer_reprojection_arms",
    "disagreement_torso_deg",
    "disagreement_arms_deg",
    "initializer_velocity_arms_deg",
    "nlf_velocity_arms_deg",
    "time_gap_reference_units",
    "valid_torso_fraction",
    "valid_arms_fraction",
    "box_score",
    "box_area_fraction",
    "torso_scale_fraction",
)


def nlf_observation_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the NLF fields that must match across external and target runs."""
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("NLF observation metadata has no settings object")
    required = {
        "model_sha256": metadata.get("model_sha256"),
        "nlf_source_commit": metadata.get("nlf_source_commit"),
        "num_aug": settings.get("num_aug"),
        "detector_threshold": settings.get("detector_threshold"),
        "selection": settings.get("selection"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Incomplete NLF observation contract: {missing}")
    return required


def geodesic_blend(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Move from ``first`` toward ``second`` on SO(3)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    delta = second @ np.swapaxes(first, -1, -2)
    tangent = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec()
    step = Rotation.from_rotvec(alpha * tangent).as_matrix().reshape(delta.shape)
    return (step @ first).astype(np.float32)


def global_rotations(local_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    result = np.empty_like(local_rotations)
    for joint, parent in enumerate(parents):
        result[joint] = (
            local_rotations[joint]
            if int(parent) < 0
            else result[int(parent)] @ local_rotations[joint]
        )
    return result


def preserve_global_rotations(
    reference_local: np.ndarray,
    candidate_local: np.ndarray,
    parents: np.ndarray,
    joint_indices: Sequence[int] = GLOBAL_WRISTS,
) -> np.ndarray:
    """Compensate wrist locals after upstream body rotations change."""
    result = candidate_local.copy()
    reference_global = global_rotations(reference_local, parents)
    for joint in joint_indices:
        parent = int(parents[joint])
        candidate_global = global_rotations(result, parents)
        result[joint] = (
            reference_global[joint]
            if parent < 0
            else candidate_global[parent].T @ reference_global[joint]
        )
    return result.astype(np.float32)


def full_initializer_rotations(cache: Any, frame: int) -> np.ndarray:
    """Expand the Phase-2 51-joint pose contract to native SMPL-X 55 joints."""
    rotvec = np.zeros((55, 3), dtype=np.float32)
    rotvec[0] = cache.global_orient[frame]
    rotvec[NLF_BODY] = cache.init_axis_angle[frame, :BODY_COUNT]
    rotvec[22] = cache.jaw_pose[frame]
    rotvec[23] = cache.leye_pose[frame]
    rotvec[24] = cache.reye_pose[frame]
    rotvec[NLF_LEFT_HAND] = cache.init_axis_angle[frame, LEFT_HAND]
    rotvec[NLF_RIGHT_HAND] = cache.init_axis_angle[frame, RIGHT_HAND]
    return Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)


def nlf_body_candidate(
    cache: Any,
    frame: int,
    nlf_pose: np.ndarray,
    parents: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse NLF body only, while preserving identity, hands and global wrists."""
    reference = full_initializer_rotations(cache, frame)
    nlf = Rotation.from_rotvec(np.asarray(nlf_pose).reshape(55, 3)).as_matrix()
    candidate = reference.copy()
    candidate[NLF_BODY] = geodesic_blend(
        reference[NLF_BODY], nlf[NLF_BODY], alpha
    )
    candidate[22:55] = reference[22:55]
    return preserve_global_rotations(reference, candidate, parents)


def cache_pose_from_full(full: np.ndarray) -> np.ndarray:
    """Return the 51 local rotations used by the Phase-2 cache contract."""
    return np.concatenate(
        (full[NLF_BODY], full[NLF_LEFT_HAND], full[NLF_RIGHT_HAND]), axis=0
    ).astype(np.float32)


def geodesic_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = first @ np.swapaxes(second, -1, -2)
    rotvec = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec()
    return np.rad2deg(np.linalg.norm(rotvec, axis=-1)).reshape(delta.shape[:-2])


def tals(values: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """Tolerance-aware loss: ignore small 2D noise and retain gross errors."""
    return np.maximum(np.asarray(values, dtype=np.float32) - tolerance, 0.0)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    return float(np.average(values[valid], weights=weights[valid]))


def _normalized_nlf_joints(observation: Any, image_size: np.ndarray) -> np.ndarray:
    height, width = np.asarray(image_size, dtype=np.float32)
    result = np.asarray(observation["joints2d"], dtype=np.float32).copy()
    result[:, 0] /= max(float(width), 1.0)
    result[:, 1] /= max(float(height), 1.0)
    return result


def _torso_scale(points: np.ndarray, valid: np.ndarray) -> float:
    # Shoulder span is the primary normalization; fall back to the torso bbox.
    if valid[11] and valid[12]:
        return float(np.linalg.norm(points[11] - points[12]))
    selected = points[TORSO[valid[TORSO]]]
    if len(selected) < 2:
        return 0.0
    return float(np.linalg.norm(selected.max(axis=0) - selected.min(axis=0)))


def frame_features(
    cache: Any,
    frame: int,
    observation: Any,
    candidate_full: np.ndarray,
    previous_initializer: np.ndarray | None = None,
    previous_nlf_body: np.ndarray | None = None,
    elapsed_seconds: float | None = None,
    reference_seconds: float = 1.0 / 15.0,
) -> dict[str, float]:
    """Build features available identically in external training and inference."""
    initial_full = full_initializer_rotations(cache, frame)
    initial_cache = cache_pose_from_full(initial_full)
    candidate_cache = cache_pose_from_full(candidate_full)
    disagreement = geodesic_degrees(candidate_cache, initial_cache)

    observed = np.asarray(cache.keypoints_2d[frame], dtype=np.float32)
    valid = np.asarray(cache.keypoint_valid[frame], dtype=bool)
    weights = np.where(valid, np.asarray(cache.u0_reliability[frame]), 0.0)
    nlf_points = _normalized_nlf_joints(observation, cache.image_size[frame])[NLF_BODY]
    residual = np.linalg.norm(nlf_points - observed[:BODY_COUNT], axis=-1)
    init_residual = np.linalg.norm(
        np.asarray(cache.reprojection_residual_2d[frame, :BODY_COUNT]), axis=-1
    )
    uncertainty = np.asarray(observation["joint_uncertainties"], dtype=np.float32)[
        NLF_BODY
    ]

    if reference_seconds <= 0:
        raise ValueError("reference_seconds must be positive")
    gap_units = 1.0
    if elapsed_seconds is not None:
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        gap_units = float(elapsed_seconds / reference_seconds)
    initializer_velocity = 0.0
    nlf_velocity = 0.0
    if previous_initializer is not None:
        initializer_velocity = float(
            geodesic_degrees(initial_cache[ARMS], previous_initializer[ARMS]).mean()
        ) / gap_units
    if previous_nlf_body is not None:
        nlf_velocity = float(
            geodesic_degrees(candidate_cache[ARMS], previous_nlf_body[ARMS]).mean()
        ) / gap_units

    box = np.asarray(observation["boxes"], dtype=np.float32).reshape(-1)
    height, width = np.asarray(cache.image_size[frame], dtype=np.float32)
    image_area = max(float(height * width), 1.0)
    values = {
        "nlf_unc_torso": float(uncertainty[TORSO].mean()),
        "nlf_unc_arms": float(uncertainty[ARMS].mean()),
        "nlf_fit2d_torso": _weighted_mean(residual[TORSO], weights[TORSO]),
        "nlf_fit2d_arms": _weighted_mean(residual[ARMS], weights[ARMS]),
        "nlf_tals2d_torso": _weighted_mean(tals(residual[TORSO]), weights[TORSO]),
        "nlf_tals2d_arms": _weighted_mean(tals(residual[ARMS]), weights[ARMS]),
        "initializer_reprojection_torso": _weighted_mean(
            init_residual[TORSO], weights[TORSO]
        ),
        "initializer_reprojection_arms": _weighted_mean(
            init_residual[ARMS], weights[ARMS]
        ),
        "disagreement_torso_deg": float(disagreement[TORSO].mean()),
        "disagreement_arms_deg": float(disagreement[ARMS].mean()),
        "initializer_velocity_arms_deg": initializer_velocity,
        "nlf_velocity_arms_deg": nlf_velocity,
        "time_gap_reference_units": gap_units,
        "valid_torso_fraction": float(valid[TORSO].mean()),
        "valid_arms_fraction": float(valid[ARMS].mean()),
        "box_score": float(box[4]) if len(box) >= 5 else 1.0,
        "box_area_fraction": float(box[2] * box[3] / image_area),
        "torso_scale_fraction": _torso_scale(observed, valid),
    }
    if tuple(values) != FEATURE_COLUMNS:
        raise AssertionError("V2 feature contract changed unexpectedly")
    return values


def external_upper_body_error_degrees(
    cache: Any, frame: int, pose: np.ndarray
) -> float:
    """External-only pseudo-target error used to train/calibrate the router."""
    if cache.target_axis_angle is None or cache.target_rotation_valid is None:
        raise ValueError("external router training requires external targets")
    valid = np.asarray(cache.target_rotation_valid[frame, :BODY_COUNT], dtype=bool)
    selected = UPPER_BODY[valid[UPPER_BODY]]
    if len(selected) < 4:
        return float("nan")
    target = Rotation.from_rotvec(cache.target_axis_angle[frame, selected]).as_matrix()
    errors = geodesic_degrees(pose[selected], target)
    arm_weight = np.where(np.isin(selected, ARMS), 2.0, 1.0)
    return float(np.average(errors, weights=arm_weight))


def viterbi_benefit_selection(
    predicted_delta: np.ndarray,
    margin: float,
    transition_penalty: float,
    transition_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Temporally coherent base/candidate choice with deterministic tie breaks."""
    delta = np.asarray(predicted_delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) == 0:
        raise ValueError("predicted_delta must be a non-empty vector")
    if transition_scales is None:
        scales = np.ones_like(delta)
    else:
        scales = np.asarray(transition_scales, dtype=np.float64)
        if scales.shape != delta.shape or not np.isfinite(scales).all():
            raise ValueError("transition_scales must be finite and match predicted_delta")
        if (scales < 0).any():
            raise ValueError("transition_scales must be non-negative")
    unary = np.stack((np.zeros_like(delta), delta + margin), axis=-1)
    cost = np.full_like(unary, np.inf)
    back = np.zeros_like(unary, dtype=np.int8)
    cost[0] = unary[0]
    for frame in range(1, len(delta)):
        for state in (0, 1):
            choices = cost[frame - 1] + transition_penalty * scales[frame] * np.asarray(
                (state != 0, state != 1), dtype=np.float64
            )
            previous = int(np.argmin(choices))
            cost[frame, state] = choices[previous] + unary[frame, state]
            back[frame, state] = previous
    states = np.zeros(len(delta), dtype=np.int8)
    states[-1] = int(np.argmin(cost[-1]))
    for frame in range(len(delta) - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return states.astype(bool)
