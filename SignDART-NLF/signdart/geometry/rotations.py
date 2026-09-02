from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def deterministic_orthogonal_axis(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    basis = np.eye(3)[int(np.argmin(np.abs(vector)))]
    axis = np.cross(vector, basis)
    return axis / max(np.linalg.norm(axis), 1e-12)


def rotation_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    first /= max(np.linalg.norm(first), 1e-12)
    second /= max(np.linalg.norm(second), 1e-12)
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return np.eye(3)
    if cosine < -1.0 + 1e-10:
        return Rotation.from_rotvec(
            np.pi * deterministic_orthogonal_axis(first)
        ).as_matrix()
    cross = np.cross(first, second)
    cross_matrix = skew(cross)
    return np.eye(3) + cross_matrix + cross_matrix @ cross_matrix / (1.0 + cosine)


def local_matrices(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    rotvec = np.concatenate(
        (
            np.asarray(global_orient, dtype=np.float64).reshape(1, 3),
            np.asarray(body_pose, dtype=np.float64).reshape(21, 3),
        ),
        axis=0,
    )
    return Rotation.from_rotvec(rotvec).as_matrix()


def global_matrices(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    output = np.empty_like(local)
    for joint, parent in enumerate(np.asarray(parents, dtype=np.int64)[: len(local)]):
        output[joint] = local[joint] if parent < 0 else output[parent] @ local[joint]
    return output


def matrices_to_body_pose(local: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(local[1:22]).as_rotvec().reshape(63).astype(np.float32)


def geodesic_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

