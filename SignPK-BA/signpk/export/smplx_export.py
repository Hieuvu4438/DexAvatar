from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from signpk.data.frame_manifest import SignManifest
from signpk.geometry.rotations import matrix_to_axis_angle
from signpk.geometry.topology import validate_topology, write_obj
from signpk.optimization.smplx_layer import SMPLXOutput
from signpk.optimization.state import SequenceState


def export_mesh_sequence(
    output_root: str | Path,
    manifest: SignManifest,
    vertices: Tensor | np.ndarray,
    faces: Tensor | np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_root = Path(output_root) / manifest.sign_name
    meshes_root = output_root / "meshes"
    vertices = torch.as_tensor(vertices)
    faces_array = torch.as_tensor(faces)
    if vertices.shape != (len(manifest.records), 10475, 3):
        raise ValueError(f"unexpected exported sequence shape {tuple(vertices.shape)}")
    for record, frame_vertices in zip(manifest.records, vertices):
        validate_topology(frame_vertices, faces_array, vertex_count=10475)
        write_obj(meshes_root / f"mesh_{record.prediction_frame_id:06d}.obj", frame_vertices, faces_array)
    manifest.save(output_root / "manifest.json")
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def export_state_sequence(
    output_root: str | Path,
    manifest: SignManifest,
    state: SequenceState,
    output: SMPLXOutput,
    faces: Tensor,
    metadata: dict[str, Any],
    vertex_transform: Tensor | None = None,
) -> None:
    vertices = output.vertices
    if vertex_transform is not None:
        transform = torch.as_tensor(vertex_transform, dtype=vertices.dtype, device=vertices.device)
        if transform.shape != (3, 3):
            raise ValueError("export vertex transform must be 3x3")
        vertices = torch.einsum("ij,tvj->tvi", transform, vertices)
    export_mesh_sequence(output_root, manifest, vertices, faces, metadata)
    parameters_root = Path(output_root) / manifest.sign_name / "parameters"
    parameters_root.mkdir(parents=True, exist_ok=True)
    rotations = state.rotations()
    for index, record in enumerate(manifest.records):
        row = {
            "betas": state.beta.detach().cpu().numpy(),
            "global_orient": matrix_to_axis_angle(rotations.root[index]).detach().cpu().numpy()[None],
            "body_pose": matrix_to_axis_angle(rotations.body[index]).flatten().detach().cpu().numpy()[None],
            "left_hand_pose": matrix_to_axis_angle(rotations.left_hand[index]).flatten().detach().cpu().numpy()[None],
            "right_hand_pose": matrix_to_axis_angle(rotations.right_hand[index]).flatten().detach().cpu().numpy()[None],
            "transl": state.translation[index].detach().cpu().numpy()[None],
            "expression": state.expression[index].detach().cpu().numpy()[None],
        }
        with (parameters_root / f"mesh_{record.prediction_frame_id:06d}.pkl").open("wb") as handle:
            pickle.dump(row, handle, protocol=pickle.HIGHEST_PROTOCOL)
