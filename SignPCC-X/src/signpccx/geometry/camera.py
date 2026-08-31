from __future__ import annotations

import numpy as np


def affine_2x3_to_homogeneous(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (2, 3):
        raise ValueError(transform.shape)
    result = np.eye(3, dtype=np.float32)
    result[:2] = transform
    return result


def transform_xy(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    destination = homogeneous @ np.asarray(transform, dtype=np.float32).T
    return destination[:, :2] / destination[:, 2:3]


def project_opencv(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam)
    intrinsic = np.asarray(intrinsic)
    if points_cam.shape[-1] != 3 or intrinsic.shape != (3, 3):
        raise ValueError((points_cam.shape, intrinsic.shape))
    z = np.maximum(points_cam[..., 2], 1e-4)
    u = intrinsic[0, 0] * points_cam[..., 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_cam[..., 1] / z + intrinsic[1, 2]
    return np.stack((u, v), axis=-1)

