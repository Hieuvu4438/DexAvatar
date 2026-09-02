from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .selector import normalized_bones


POINTMAP_CHAIN_IDS = {
    "left": (16, 18, 20),
    "right": (17, 19, 21),
}
PART_JOINT_IDS = {
    "left_upper": 16,
    "left_forearm": 18,
    "right_upper": 17,
    "right_forearm": 19,
}
PART_ENDPOINT_IDS = {
    "left_upper": (16, 18),
    "left_forearm": (18, 20),
    "right_upper": (17, 19),
    "right_forearm": (19, 21),
}


def face_part_labels(weights: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Assign arm faces from the fixed dominant SMPL-X skinning joint."""
    dominant = np.asarray(weights).argmax(axis=1)
    face_joints = dominant[np.asarray(faces, dtype=np.int64)]
    labels = np.full(len(faces), -1, dtype=np.int16)
    for part_index, joint_id in enumerate(PART_JOINT_IDS.values()):
        labels[(face_joints == joint_id).sum(axis=1) >= 2] = part_index
    return labels


def render_visible_part_masks(
    vertices_evaluator: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    evaluator_K: np.ndarray,
    height: int,
    width: int,
    erode_px: int = 2,
) -> dict[str, np.ndarray]:
    """Render fixed SMPL-X part labels with a depth-sorted triangle painter."""
    vertices = np.asarray(vertices_evaluator, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    homogeneous = vertices @ np.asarray(evaluator_K, dtype=np.float64).T
    uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    positive_depth = -vertices[:, 2]
    face_depth = positive_depth[faces].mean(axis=1)
    valid = np.isfinite(uv[faces]).all(axis=(1, 2))
    valid &= (positive_depth[faces] > 1e-6).all(axis=1)
    order = np.flatnonzero(valid)[np.argsort(face_depth[valid])[::-1]]
    canvas = np.zeros((height, width), dtype=np.uint8)
    for face_index in order:
        triangle = np.rint(uv[faces[face_index]] * 16.0).astype(np.int32)
        value = int(labels[face_index]) + 1 if labels[face_index] >= 0 else 0
        cv2.fillConvexPoly(canvas, triangle, value, lineType=cv2.LINE_8, shift=4)
    kernel = np.ones((3, 3), dtype=np.uint8)
    masks = {}
    for index, name in enumerate(PART_JOINT_IDS):
        mask = (canvas == index + 1).astype(np.uint8)
        if erode_px > 0:
            mask = cv2.erode(mask, kernel, iterations=erode_px)
        masks[name] = mask.astype(bool)
    return masks


def mask_bone_endpoints(
    upper_mask: np.ndarray,
    forearm_mask: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Infer directed 2D bone endpoints from adjacent semantic arm regions."""
    masks = {
        "upper": np.asarray(upper_mask, dtype=bool),
        "forearm": np.asarray(forearm_mask, dtype=bool),
    }
    points = {}
    centers = {}
    endpoints = {}
    for name, mask in masks.items():
        yy, xx = np.nonzero(mask)
        if len(xx) < 8:
            raise ValueError(f"too few {name} mask pixels")
        cloud = np.stack((xx, yy), axis=1).astype(np.float64)
        center = cloud.mean(axis=0)
        _, _, right = np.linalg.svd(cloud - center, full_matrices=False)
        axis = right[0]
        coordinate = (cloud - center) @ axis
        low, high = np.quantile(coordinate, [0.05, 0.95])
        points[name] = cloud
        centers[name] = center
        endpoints[name] = (center + low * axis, center + high * axis)
    upper_ends = endpoints["upper"]
    forearm_ends = endpoints["forearm"]
    upper_elbow_index = int(
        np.argmin([np.linalg.norm(value - centers["forearm"]) for value in upper_ends])
    )
    forearm_elbow_index = int(
        np.argmin([np.linalg.norm(value - centers["upper"]) for value in forearm_ends])
    )
    shoulder = upper_ends[1 - upper_elbow_index]
    upper_elbow = upper_ends[upper_elbow_index]
    forearm_elbow = forearm_ends[forearm_elbow_index]
    wrist = forearm_ends[1 - forearm_elbow_index]
    elbow = 0.5 * (upper_elbow + forearm_elbow)
    return {
        "upper": (shoulder, elbow),
        "forearm": (elbow, wrist),
    }


def _orient_axis(
    axis: np.ndarray,
    points: np.ndarray,
    pixels: np.ndarray,
    center: np.ndarray,
    uv_parent: np.ndarray,
    uv_child: np.ndarray,
) -> np.ndarray:
    direction_2d = np.asarray(uv_child) - np.asarray(uv_parent)
    denominator = float(direction_2d @ direction_2d)
    if denominator < 1e-8:
        raise ValueError("degenerate projected bone")
    progress = ((pixels - np.asarray(uv_parent)) @ direction_2d) / denominator
    coordinate = (points - center) @ axis
    correlation = np.corrcoef(progress, coordinate)[0, 1]
    if not np.isfinite(correlation):
        raise ValueError("undefined 2D/3D axis orientation")
    return -axis if correlation < 0.0 else axis


def robust_axis(
    points_xyz: np.ndarray,
    pixels_uv: np.ndarray,
    uv_parent: np.ndarray,
    uv_child: np.ndarray,
    iterations: int = 5,
) -> tuple[np.ndarray, dict]:
    points = np.asarray(points_xyz, dtype=np.float64)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("too few pointmap pixels")
    weights = np.ones(len(points), dtype=np.float64)
    center = np.zeros(3)
    axis = np.zeros(3)
    eigenvalues = np.zeros(3)
    mad = np.inf
    for _ in range(iterations):
        normalized = weights / max(float(weights.sum()), 1e-12)
        center = (normalized[:, None] * points).sum(axis=0)
        centered = points - center
        covariance = (normalized[:, None] * centered).T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        residual = np.linalg.norm(np.cross(centered, axis[None]), axis=1)
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median))) + 1e-9
        z = residual / (4.685 * mad)
        weights = np.square(1.0 - np.square(z))
        weights[z >= 1.0] = 0.0
        if weights.sum() < 3.0:
            weights.fill(1.0)
    axis = _orient_axis(axis, points, pixels, center, uv_parent, uv_child)
    gap = float(
        (eigenvalues[-1] - eigenvalues[-2]) / max(eigenvalues[-1], 1e-12)
    )
    return axis, {
        "n": int(len(points)),
        "eigen_gap": gap,
        "residual_mad": float(mad),
    }


def block_bootstrap_axes(
    points_xyz: np.ndarray,
    pixels_uv: np.ndarray,
    uv_parent: np.ndarray,
    uv_child: np.ndarray,
    reference_axis: np.ndarray,
    seed_key: str,
    repetitions: int = 256,
    grid_size: int = 4,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    minimum = pixels.min(axis=0)
    extent = np.maximum(pixels.max(axis=0) - minimum, 1.0)
    cells = np.floor((pixels - minimum) / extent * grid_size).astype(np.int64)
    cells = np.clip(cells, 0, grid_size - 1)
    cell_id = cells[:, 1] * grid_size + cells[:, 0]
    blocks = [np.flatnonzero(cell_id == value) for value in np.unique(cell_id)]
    blocks = [block for block in blocks if len(block)]
    if len(blocks) < 2:
        raise ValueError("too few spatial blocks for bootstrap")
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(repetitions):
        sampled_blocks = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in sampled_blocks])
        if len(indices) < 3:
            continue
        try:
            axis, _ = robust_axis(
                points[indices], pixels[indices], uv_parent, uv_child, iterations=3
            )
        except ValueError:
            continue
        if float(axis @ reference_axis) < 0.0:
            axis = -axis
        output.append(axis)
    if len(output) < max(32, repetitions // 2):
        raise ValueError("insufficient valid bootstrap replicates")
    return np.asarray(output, dtype=np.float32)


def pointmap_candidate_energy(
    candidate_joints: np.ndarray,
    axes: np.ndarray,
    reliability: np.ndarray,
    side: str,
    huber_delta_deg: float = 10.0,
) -> np.ndarray:
    ids = POINTMAP_CHAIN_IDS[side]
    directions = np.stack([
        normalized_bones(joints, ids) for joints in candidate_joints
    ])
    cosine = np.clip(np.einsum("nbd,bd->nb", directions, axes), -1.0, 1.0)
    angle = np.arccos(cosine)
    delta = np.deg2rad(huber_delta_deg)
    loss = np.where(
        angle <= delta,
        0.5 * np.square(angle),
        delta * (angle - 0.5 * delta),
    )
    weights = np.asarray(reliability, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.sum(loss * weights[None], axis=1)


def pointmap_bootstrap_decision(
    candidate_joints: np.ndarray,
    axes: np.ndarray,
    bootstrap_axes: np.ndarray,
    reliability: np.ndarray,
    side: str,
    incumbent_index: int = 0,
) -> tuple[int, dict]:
    energy = pointmap_candidate_energy(
        candidate_joints, axes, reliability, side
    )
    best = int(np.argmin(energy))
    diagnostics = {
        "incumbent_energy": float(energy[incumbent_index]),
        "best_energy": float(energy[best]),
        "best_index": best,
        "reason": "retain_incumbent",
    }
    if best == incumbent_index:
        diagnostics["reason"] = "incumbent_preferred"
        return incumbent_index, diagnostics
    repetitions = min(len(item) for item in bootstrap_axes)
    if repetitions < 32:
        diagnostics["reason"] = "insufficient_bootstrap_axes"
        return incumbent_index, diagnostics
    differences = []
    for replicate in range(repetitions):
        replicate_axes = np.stack([
            bootstrap_axes[bone][replicate] for bone in range(2)
        ])
        replicate_energy = pointmap_candidate_energy(
            candidate_joints, replicate_axes, reliability, side
        )
        differences.append(
            float(replicate_energy[best] - replicate_energy[incumbent_index])
        )
    differences = np.asarray(differences)
    upper_confidence = float(np.quantile(differences, 0.95))
    diagnostics.update({
        "bootstrap_repetitions": int(repetitions),
        "energy_difference_mean": float(differences.mean()),
        "energy_difference_upper95": upper_confidence,
    })
    if upper_confidence >= 0.0:
        diagnostics["reason"] = "incumbent_not_beaten_with_confidence"
        return incumbent_index, diagnostics
    diagnostics["reason"] = "pointmap_branch_accepted"
    return best, diagnostics
