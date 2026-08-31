"""Pinned differentiable SMPL-X forward adapter for DCG trajectory state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from dcg_sign4d.diffusion.state_codec import TrajectoryState, rotation_6d_to_matrix
from dcg_sign4d.utils.hashing import file_sha256

from .so3 import log_map


@dataclass(frozen=True)
class SMPLXForwardOutput:
    vertices: Tensor
    joints: Tensor


class SMPLXAdapter(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_sha256: str,
        trusted_model: bool,
        coordinate_convention: str = "dexavatar_camera_x_180",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        path = Path(model_path)
        if not trusted_model:
            raise PermissionError("licensed SMPL-X model loading requires trusted_model=True")
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"SMPL-X model hash mismatch: {path}")
        if coordinate_convention not in {"native", "dexavatar_camera_x_180"}:
            raise ValueError("unsupported coordinate convention")
        try:
            import smplx
        except ImportError as exc:
            raise RuntimeError("install the licensed smplx runtime") from exc
        self.model_path = path
        self.model_hash = expected_sha256
        self.coordinate_convention = coordinate_convention
        self.model = smplx.SMPLX(
            str(path),
            gender="neutral",
            ext=path.suffix.lstrip("."),
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10,
            dtype=dtype,
        )

    def forward(self, state: TrajectoryState) -> SMPLXForwardOutput:
        state.validate()
        batch, time = state.root_translation.shape[:2]
        count = batch * time

        def axis_angle(rotation: Tensor) -> Tensor:
            return log_map(rotation_6d_to_matrix(rotation)).reshape(count, -1)

        if state.beta.shape[-1] != 10:
            raise ValueError("this pinned SMPL-X adapter requires 10 beta coefficients")
        if state.body_rot6d.shape[2] != 21:
            raise ValueError("SMPL-X body topology requires 21 body joints")
        if state.left_hand_rot6d.shape[2:] != (15, 6) or state.right_hand_rot6d.shape[2:] != (
            15,
            6,
        ):
            raise ValueError("SMPL-X hand topology requires 15 joints per hand")
        zeros = state.root_translation.new_zeros(count, 3)
        expression = state.root_translation.new_zeros(count, 10)
        jaw, left_eye, right_eye = zeros, zeros, zeros
        if state.face_state is not None:
            if state.face_state.shape[-1] != 19:
                raise ValueError("face_state must be jaw/eyes/expression [19]")
            face = state.face_state.reshape(count, 19)
            jaw, left_eye, right_eye, expression = face.split((3, 3, 3, 10), dim=-1)
        output = self.model(
            global_orient=axis_angle(state.root_rot6d),
            body_pose=axis_angle(state.body_rot6d),
            left_hand_pose=axis_angle(state.left_hand_rot6d),
            right_hand_pose=axis_angle(state.right_hand_rot6d),
            jaw_pose=jaw,
            leye_pose=left_eye,
            reye_pose=right_eye,
            betas=state.beta[:, None, :].expand(batch, time, -1).reshape(count, 10),
            expression=expression,
            transl=state.root_translation.reshape(count, 3),
            pose2rot=True,
            return_verts=True,
        )
        vertices = output.vertices.reshape(batch, time, -1, 3)
        joints = output.joints.reshape(batch, time, -1, 3)
        if self.coordinate_convention == "dexavatar_camera_x_180":
            camera_x_180 = vertices.new_tensor([1.0, -1.0, -1.0])
            vertices = vertices * camera_x_180
            joints = joints * camera_x_180
        return SMPLXForwardOutput(vertices, joints)
