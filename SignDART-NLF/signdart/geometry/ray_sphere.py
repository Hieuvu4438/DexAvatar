from __future__ import annotations

import numpy as np


def pixel_ray(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Return the z=1 camera ray for a pixel under an exact intrinsic matrix."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    ray = np.linalg.solve(K, np.asarray([uv[0], uv[1], 1.0]))
    if not np.isfinite(ray).all() or abs(ray[2]) < 1e-12:
        raise ValueError("invalid pixel ray")
    return ray / ray[2]


def project(K: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.asarray(K, dtype=np.float64) @ np.asarray(point, dtype=np.float64)
    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-12:
        raise ValueError("point cannot be projected")
    return homogeneous[:2] / homogeneous[2]


def positive_sphere_ray_roots(
    parent: np.ndarray,
    ray: np.ndarray,
    length: float,
    *,
    disc_eps: float = 1e-10,
) -> list[np.ndarray]:
    """Solve ``||lambda ray - parent|| = length`` for positive depth."""
    parent = np.asarray(parent, dtype=np.float64).reshape(3)
    ray = np.asarray(ray, dtype=np.float64).reshape(3)
    length = float(length)
    a = float(ray @ ray)
    b = float(-2.0 * (ray @ parent))
    c = float(parent @ parent - length * length)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -disc_eps:
        return []
    root = np.sqrt(max(discriminant, 0.0))
    values = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    output: list[np.ndarray] = []
    for depth in values:
        point = depth * ray
        if depth <= 1e-8 or not np.isfinite(point).all():
            continue
        if not output or np.linalg.norm(point - output[0]) > 1e-8:
            output.append(point)
    return output


def enumerate_arm_branches(
    shoulder: np.ndarray,
    elbow_uv: np.ndarray,
    wrist_uv: np.ndarray,
    upper_length: float,
    forearm_length: float,
    K: np.ndarray,
) -> list[dict[str, np.ndarray | str]]:
    elbow_ray = pixel_ray(K, elbow_uv)
    wrist_ray = pixel_ray(K, wrist_uv)
    candidates: list[dict[str, np.ndarray | str]] = []
    for elbow_index, elbow in enumerate(
        positive_sphere_ray_roots(shoulder, elbow_ray, upper_length)
    ):
        for wrist_index, wrist in enumerate(
            positive_sphere_ray_roots(elbow, wrist_ray, forearm_length)
        ):
            candidates.append(
                {
                    "name": f"e{elbow_index}_w{wrist_index}",
                    "shoulder": np.asarray(shoulder, dtype=np.float64),
                    "elbow": elbow,
                    "wrist": wrist,
                }
            )
    return candidates


def enumerate_three_link_branches(
    collar: np.ndarray,
    shoulder_uv: np.ndarray,
    elbow_uv: np.ndarray,
    wrist_uv: np.ndarray,
    collar_length: float,
    upper_length: float,
    forearm_length: float,
    K: np.ndarray,
) -> list[dict[str, np.ndarray | str]]:
    """Enumerate the finite collar--shoulder--elbow--wrist depth tree."""
    shoulder_ray = pixel_ray(K, shoulder_uv)
    elbow_ray = pixel_ray(K, elbow_uv)
    wrist_ray = pixel_ray(K, wrist_uv)
    candidates: list[dict[str, np.ndarray | str]] = []
    for shoulder_index, shoulder in enumerate(
        positive_sphere_ray_roots(collar, shoulder_ray, collar_length)
    ):
        for elbow_index, elbow in enumerate(
            positive_sphere_ray_roots(shoulder, elbow_ray, upper_length)
        ):
            for wrist_index, wrist in enumerate(
                positive_sphere_ray_roots(elbow, wrist_ray, forearm_length)
            ):
                candidates.append(
                    {
                        "name": (
                            f"s{shoulder_index}_e{elbow_index}_w{wrist_index}"
                        ),
                        "shoulder": shoulder,
                        "elbow": elbow,
                        "wrist": wrist,
                    }
                )
    return candidates
