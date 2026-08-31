from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from signpk.geometry.rotations import matrix_to_axis_angle
from signpk.utils.config_hash import sha256_file

from .state import SequenceState


@dataclass
class SMPLXOutput:
    vertices: Tensor
    joints: Tensor
    left_hand_joints: Tensor
    right_hand_joints: Tensor


def _make_hand_regressors(model_data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    regressor = model_data["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor)
    vertex_count = 10475
    def tip(index: int) -> np.ndarray:
        row = np.zeros((1, vertex_count), dtype=np.float32)
        row[0, index] = 1.0
        return row
    # Output order: wrist, thumb1..tip, index1..tip, middle, ring, pinky.
    left = np.concatenate(
        [
            regressor[[20]], regressor[[37, 38, 39]], tip(5361),
            regressor[[25, 26, 27]], tip(4933),
            regressor[[28, 29, 30]], tip(5058),
            regressor[[34, 35, 36]], tip(5169),
            regressor[[31, 32, 33]], tip(5286),
        ],
        axis=0,
    )
    right = np.concatenate(
        [
            regressor[[21]], regressor[[52, 53, 54]], tip(8079),
            regressor[[40, 41, 42]], tip(7669),
            regressor[[43, 44, 45]], tip(7794),
            regressor[[49, 50, 51]], tip(7905),
            regressor[[46, 47, 48]], tip(8022),
        ],
        axis=0,
    )
    return left.astype(np.float32), right.astype(np.float32)


class SMPLXLayer(nn.Module):
    """Pinned, differentiable standard-topology SMPL-X boundary."""

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str | None = None,
        output_transform: Tensor | None = None,
    ):
        super().__init__()
        model_path = Path(model_path)
        if expected_sha256 and sha256_file(model_path) != expected_sha256:
            raise ValueError(f"SMPL-X model hash mismatch: {model_path}")
        try:
            import smplx
        except ImportError as exc:
            raise RuntimeError("the licensed smplx package is required for fitting") from exc
        self.model_hash = sha256_file(model_path)
        self.model = smplx.SMPLX(
            str(model_path),
            gender="neutral",
            ext=model_path.suffix.lstrip("."),
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with np.load(model_path, allow_pickle=True) as data:
            left, right = _make_hand_regressors(data)
            faces = np.asarray(data["f"], dtype=np.int64)
        self.register_buffer("left_hand_regressor", torch.from_numpy(left))
        self.register_buffer("right_hand_regressor", torch.from_numpy(right))
        self.register_buffer("faces", torch.from_numpy(faces))
        transform = torch.eye(3) if output_transform is None else torch.as_tensor(output_transform, dtype=torch.float32)
        if transform.shape != (3, 3):
            raise ValueError("output transform must be 3x3")
        self.register_buffer("output_transform", transform)

    def forward(self, state: SequenceState) -> SMPLXOutput:
        rotations = state.rotations()
        frames = state.num_frames
        zeros = state.translation.new_zeros((frames, 3))
        result = self.model(
            global_orient=matrix_to_axis_angle(rotations.root),
            body_pose=matrix_to_axis_angle(rotations.body).flatten(1),
            left_hand_pose=matrix_to_axis_angle(rotations.left_hand).flatten(1),
            right_hand_pose=matrix_to_axis_angle(rotations.right_hand).flatten(1),
            jaw_pose=zeros,
            leye_pose=zeros,
            reye_pose=zeros,
            betas=state.beta.expand(frames, -1),
            expression=state.expression,
            transl=state.translation,
            return_verts=True,
        )
        vertices = torch.einsum("ij,tvj->tvi", self.output_transform, result.vertices)
        joints = torch.einsum("ij,tkj->tki", self.output_transform, result.joints)
        return SMPLXOutput(
            vertices=vertices,
            joints=joints,
            left_hand_joints=torch.einsum("jv,tvc->tjc", self.left_hand_regressor, vertices),
            right_hand_joints=torch.einsum("jv,tvc->tjc", self.right_hand_regressor, vertices),
        )
