from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def write_dexavatar_obj(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    decimals: int = 8,
    face_block: str | None = None,
) -> None:
    """Write the exact simple OBJ dialect emitted by DexAvatar's Trimesh exporter."""
    target = Path(path)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
        raise ValueError(f"vertices must be finite [10475,3], got {vertices.shape}")
    if faces.shape != (20908, 3) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"faces must be valid zero-indexed [20908,3], got {faces.shape}")
    if decimals < 1 or decimals > 16:
        raise ValueError("OBJ decimal precision must be in [1,16]")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# https://github.com/mikedh/trimesh\n")
            np.savetxt(handle, vertices, fmt=f"v %.{decimals}f %.{decimals}f %.{decimals}f")
            if face_block is None:
                np.savetxt(handle, faces.astype(np.int64) + 1, fmt="f %d %d %d")
            else:
                handle.write(face_block)
                handle.write("\n")
            handle.write("\n")
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def read_simple_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value.split("/", 1)[0]) - 1 for value in line.split()[1:4]])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def format_face_block(faces: np.ndarray) -> str:
    values = np.asarray(faces, dtype=np.int64) + 1
    return "\n".join(f"f {first} {second} {third}" for first, second, third in values)
