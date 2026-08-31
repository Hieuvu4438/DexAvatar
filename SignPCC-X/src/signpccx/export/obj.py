from __future__ import annotations

import os
from pathlib import Path

import numpy as np


EXPORT_TRANSFORMS = {
    "identity": np.eye(3, dtype=np.float32),
    "x180": np.diag([1.0, -1.0, -1.0]).astype(np.float32),
    "x_180": np.diag([1.0, -1.0, -1.0]).astype(np.float32),
}


def validate_mesh(vertices: np.ndarray, faces: np.ndarray, vertex_count: int = 10475) -> None:
    if vertices.shape != (vertex_count, 3):
        raise ValueError(f"vertices {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces {faces.shape}")
    if vertices.dtype.kind != "f" or not np.isfinite(vertices).all():
        raise FloatingPointError("vertices contain NaN/Inf or are not float")
    if faces.dtype.kind not in "iu":
        raise TypeError(faces.dtype)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise IndexError((int(faces.min()), int(faces.max())))


def write_obj_atomic(
    path: Path,
    vertices_source: np.ndarray,
    canonical_faces: np.ndarray,
    transform: str = "x180",
) -> None:
    if transform not in EXPORT_TRANSFORMS:
        raise ValueError(f"Unknown export transform: {transform}")
    vertices = np.asarray(vertices_source, dtype=np.float32) @ EXPORT_TRANSFORMS[transform].T
    faces = np.asarray(canonical_faces, dtype=np.int64)
    validate_mesh(vertices, faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for i, j, k in faces + 1:
            handle.write(f"f {i:d} {j:d} {k:d}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

