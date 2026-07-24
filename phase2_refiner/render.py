"""Render standard Phase 2 result PKLs to SMPL-X OBJ meshes."""

from __future__ import annotations

import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import torch


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(path.name)
    return int(match.group(1))


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in faces + 1:
            handle.write(f"f {a} {b} {c}\n")


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append(
                    [int(value.split("/")[0]) - 1 for value in line.split()[1:4]]
                )
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int64)


def create_smplx_model(model_folder: str | Path, device: torch.device):
    from dexavatar_fitting.rewrite_body_model import SMPLX

    model_path = Path(model_folder).resolve()
    if model_path.is_dir():
        model_path = model_path / "smplx" / "SMPLX_NEUTRAL.pkl"
    return (
        SMPLX(
            model_path=str(model_path),
            ext="pkl",
            use_face_contour=True,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10,
            create_body_pose=True,
        )
        .to(device)
        .eval()
    )


def _params_to_vertices(model, params: dict, device: torch.device) -> np.ndarray:
    allowed = {
        "betas",
        "global_orient",
        "body_pose",
        "transl",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
    }
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
        for key, value in params.items()
        if key in allowed
    }
    output = model(return_verts=True, **tensors)
    vertices = output.vertices[0].detach().cpu().numpy()
    vertices[:, 1:3] *= -1.0
    return vertices


@torch.no_grad()
def render_result_directory(
    result_dir: str | Path,
    mesh_dir: str | Path,
    model_folder: str | Path,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> int:
    device = torch.device(device)
    model = create_smplx_model(model_folder, device)
    result_paths = sorted(Path(result_dir).glob("*.pkl"), key=_frame_number)
    targets = [Path(mesh_dir) / f"{path.stem}.obj" for path in result_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} meshes; first: {existing[0]}"
        )
    count = 0
    for result_path in result_paths:
        with result_path.open("rb") as handle:
            params = pickle.load(handle, encoding="latin1")
        vertices = _params_to_vertices(model, params, device)
        _write_obj(Path(mesh_dir) / f"{result_path.stem}.obj", vertices, model.faces)
        count += 1
    return count


@torch.no_grad()
def render_source_anchored_directory(
    result_dir: str | Path,
    mesh_dir: str | Path,
    source_paths: list[str] | np.ndarray,
    model_folder: str | Path,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> int:
    """Apply Phase 2 model-space vertex deltas to the existing baseline meshes.

    The checked-in fitter's saved PKLs and meshes are not perfectly reproducible
    by a fresh SMPL-X forward pass. Using a same-model output-minus-input delta
    preserves the exact existing mesh when the Phase 2 residual is zero.
    """
    device = torch.device(device)
    model = None
    result_paths = sorted(Path(result_dir).glob("*.pkl"), key=_frame_number)
    targets = [Path(mesh_dir) / f"{path.stem}.obj" for path in result_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} meshes; first: {existing[0]}"
        )
    source_by_name = {Path(path).stem: Path(path) for path in source_paths}
    count = 0
    for result_path in result_paths:
        source_result = source_by_name.get(result_path.stem)
        if source_result is None or not source_result.exists():
            raise FileNotFoundError(f"Missing source result for {result_path.name}")
        source_mesh = source_result.parent.parent / "meshes" / f"{result_path.stem}.obj"
        if not source_mesh.exists():
            raise FileNotFoundError(f"Missing source mesh: {source_mesh}")
        with source_result.open("rb") as handle:
            source_params = pickle.load(handle, encoding="latin1")
        with result_path.open("rb") as handle:
            refined_params = pickle.load(handle, encoding="latin1")
        pose_keys = (
            "betas",
            "global_orient",
            "body_pose",
            "transl",
            "left_hand_pose",
            "right_hand_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "expression",
        )
        if all(
            np.array_equal(
                np.asarray(source_params[key]), np.asarray(refined_params[key])
            )
            for key in pose_keys
        ):
            target_mesh = Path(mesh_dir) / f"{result_path.stem}.obj"
            target_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_mesh, target_mesh)
            count += 1
            continue
        if model is None:
            model = create_smplx_model(model_folder, device)
        source_vertices, source_faces = _read_obj(source_mesh)
        base_vertices = _params_to_vertices(model, source_params, device)
        refined_vertices = _params_to_vertices(model, refined_params, device)
        if source_vertices.shape != base_vertices.shape:
            raise ValueError(f"Vertex mismatch for {source_mesh}")
        vertices = source_vertices + (refined_vertices - base_vertices)
        _write_obj(Path(mesh_dir) / f"{result_path.stem}.obj", vertices, source_faces)
        count += 1
    return count
