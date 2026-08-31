from __future__ import annotations

import torch

from signpk.geometry.rotations import matrix_to_axis_angle
from signpk.optimization.smplx_layer import SMPLXLayer, SMPLXOutput

from .explicit_tokens import ExplicitTokenBatch, UPPER_BODY_INDICES
from .palm_kinematic_coupler import PKCOutput


def decode_pkc_center(
    prediction: PKCOutput,
    window: ExplicitTokenBatch,
    base_body_rotmat: torch.Tensor,
    betas: torch.Tensor,
    translation: torch.Tensor,
    decoder: SMPLXLayer,
) -> SMPLXOutput:
    """Differentiably decode a batched PKC center frame to standard SMPL-X."""

    batch = prediction.upper_rotmat.shape[0]
    center = window.timestamps.shape[1] // 2
    body = base_body_rotmat.clone()
    combined = torch.cat([window.upper_base_rotmat[:, center, :1], body], dim=1)
    combined[:, UPPER_BODY_INDICES] = prediction.upper_rotmat
    zeros = translation.new_zeros((batch, 3))
    output = decoder.model(
        global_orient=matrix_to_axis_angle(combined[:, 0]),
        body_pose=matrix_to_axis_angle(combined[:, 1:]).flatten(1),
        left_hand_pose=matrix_to_axis_angle(prediction.left_rotmat).flatten(1),
        right_hand_pose=matrix_to_axis_angle(prediction.right_rotmat).flatten(1),
        jaw_pose=zeros,
        leye_pose=zeros,
        reye_pose=zeros,
        betas=betas,
        expression=translation.new_zeros((batch, 10)),
        transl=translation,
        return_verts=True,
    )
    vertices = torch.einsum("ij,bvj->bvi", decoder.output_transform, output.vertices)
    joints = torch.einsum("ij,bkj->bki", decoder.output_transform, output.joints)
    return SMPLXOutput(
        vertices=vertices,
        joints=joints,
        left_hand_joints=torch.einsum("jv,bvc->bjc", decoder.left_hand_regressor, vertices),
        right_hand_joints=torch.einsum("jv,bvc->bjc", decoder.right_hand_regressor, vertices),
    )

