from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


SMPLX_VERTEX_COUNT = 10475


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                values = line.split()[1:]
                if len(values) != 3:
                    raise ValueError(f"non-triangular OBJ face in {path}")
                faces.append([int(value.split("/")[0]) - 1 for value in values])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_obj(path: str | Path, vertices: np.ndarray | Tensor, faces: np.ndarray | Tensor) -> None:
    vertices_np = torch.as_tensor(vertices).detach().cpu().numpy()
    faces_np = torch.as_tensor(faces).detach().cpu().numpy()
    validate_topology(vertices_np, faces_np, vertex_count=vertices_np.shape[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices_np:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces_np:
            handle.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")


def validate_topology(
    vertices: np.ndarray | Tensor,
    faces: np.ndarray | Tensor,
    reference_faces: np.ndarray | Tensor | None = None,
    vertex_count: int = SMPLX_VERTEX_COUNT,
) -> None:
    vertices_np = torch.as_tensor(vertices).detach().cpu().numpy()
    faces_np = torch.as_tensor(faces).detach().cpu().numpy()
    if vertices_np.shape != (vertex_count, 3):
        raise ValueError(f"expected {(vertex_count, 3)} vertices, got {vertices_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if not np.isfinite(vertices_np).all():
        raise ValueError("mesh contains NaN/Inf")
    if faces_np.min(initial=0) < 0 or faces_np.max(initial=-1) >= vertex_count:
        raise ValueError("face index is outside the vertex array")
    if reference_faces is not None:
        np.testing.assert_array_equal(faces_np, np.asarray(reference_faces))


def load_reference_faces(smplx_npz: str | Path) -> np.ndarray:
    with np.load(Path(smplx_npz), allow_pickle=True) as model:
        faces = np.asarray(model["f"], dtype=np.int64)
    return faces
