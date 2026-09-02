from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .geometry.arm_ik import BOUNDARY_X180
from .io.h1_state import H1State, STATE_KEYS


def create_model(model_root: Path, device: str):
    import smplx

    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        use_pca=False,
        use_face_contour=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def forward_state_batch(
    model,
    state: H1State,
    body_poses: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    body_poses = np.asarray(body_poses, dtype=np.float32).reshape(-1, 63)
    batch = body_poses.shape[0]
    kwargs = {}
    for key in STATE_KEYS:
        value = body_poses if key == "body_pose" else np.repeat(state.arrays[key], batch, axis=0)
        kwargs[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(**kwargs, return_verts=True)
    vertices_internal = output.vertices.detach().cpu().numpy()
    joints_internal = output.joints.detach().cpu().numpy()
    boundary = np.diag(BOUNDARY_X180).astype(np.float32)
    return vertices_internal * boundary, joints_internal


def rigid_transport_hand_vertices(
    candidate_vertices_evaluator: np.ndarray,
    candidate_joints_internal: np.ndarray,
    incumbent_vertices_evaluator: np.ndarray,
    incumbent_joints_internal: np.ndarray,
    hand_ids: np.ndarray,
    wrist_id: int,
) -> np.ndarray:
    """Transport the validated H1 hand surface with the candidate wrist.

    The wrist global orientation is invariant by construction, so the only
    intended rigid motion of the distal hand is the wrist translation. This
    correction removes ancestor-weight leakage from SMPL-X linear blend
    skinning while retaining canonical topology and vertex order.
    """
    output = np.asarray(candidate_vertices_evaluator).copy()
    boundary = np.diag(BOUNDARY_X180).astype(np.float32)
    displacement = (
        np.asarray(candidate_joints_internal[wrist_id])
        - np.asarray(incumbent_joints_internal[wrist_id])
    ) * boundary
    output[np.asarray(hand_ids, dtype=np.int64)] = (
        np.asarray(incumbent_vertices_evaluator)[np.asarray(hand_ids, dtype=np.int64)]
        + displacement
    )
    return output
