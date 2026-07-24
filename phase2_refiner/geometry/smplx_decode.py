"""Differentiable SMPL-X decoding helpers for optional geometry supervision."""

from __future__ import annotations

from typing import Any

import torch

from phase2_refiner.geometry.rotations import matrix_to_axis_angle


def split_pose_matrices(matrix: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Convert [B,T,51,3,3] matrices to flattened SMPL-X pose fields."""
    if matrix.shape[-3:] != (51, 3, 3):
        raise ValueError(f"Expected [B,T,51,3,3], got {tuple(matrix.shape)}")
    axis_angle = matrix_to_axis_angle(matrix)
    return (
        axis_angle[..., :21, :].flatten(-2),
        axis_angle[..., 21:36, :].flatten(-2),
        axis_angle[..., 36:51, :].flatten(-2),
    )


def decode_smplx_sequence(
    model: Any,
    matrix: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    **face: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a sequence while preserving gradients to the pose matrices."""
    body, left, right = split_pose_matrices(matrix)
    batch, frames = body.shape[:2]

    def flatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch * frames, -1)

    shared_betas = betas[:, None].expand(-1, frames, -1)
    output = model(
        body_pose=flatten(body),
        left_hand_pose=flatten(left),
        right_hand_pose=flatten(right),
        betas=flatten(shared_betas),
        global_orient=flatten(global_orient),
        transl=flatten(transl),
        return_verts=True,
        **{name: flatten(value) for name, value in face.items()},
    )
    vertices = output.vertices.reshape(batch, frames, -1, 3)
    joints = output.joints.reshape(batch, frames, -1, 3)
    return vertices, joints
