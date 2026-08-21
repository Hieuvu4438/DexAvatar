from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..geometry.so3 import log_map
from ..optimization.state import SequenceState
from ..utils.hashing import sha256_file


@dataclass
class SMPLXOutput:
    vertices: torch.Tensor
    joints: torch.Tensor


class SMPLXWrapper(nn.Module):
    """Pinned SMPL-X boundary. Model bytes remain outside the SIGNAL-4D package."""

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str | None = None,
        gender: str = "neutral",
        num_betas: int = 10,
        num_expression: int = 10,
    ) -> None:
        super().__init__()
        model_path = Path(model_path)
        if expected_sha256 and sha256_file(model_path) != expected_sha256:
            raise ValueError(f"SMPL-X hash mismatch: {model_path}")
        try:
            import smplx
        except ImportError as exc:
            raise RuntimeError(
                "Install the locally licensed smplx package to use SMPLXWrapper"
            ) from exc
        self.model_path = model_path
        self.model_hash = sha256_file(model_path)
        self.model = smplx.SMPLX(
            str(model_path),
            gender=gender,
            ext=model_path.suffix.lstrip("."),
            use_pca=False,
            flat_hand_mean=True,
            num_betas=num_betas,
            num_expression_coeffs=num_expression,
        )

    def forward(self, state: SequenceState) -> SMPLXOutput:
        rotations = state.rotations()
        t = state.translation.shape[0]
        jaw = state.translation.new_zeros((t, 3))
        eyes = state.translation.new_zeros((t, 3))
        expression = (
            state.expression
            if state.expression is not None
            else state.translation.new_zeros((t, self.model.num_expression_coeffs))
        )
        output = self.model(
            global_orient=log_map(rotations["global_orient"]),
            body_pose=log_map(rotations["body_pose"]).flatten(1),
            left_hand_pose=log_map(rotations["left_hand_pose"]).flatten(1),
            right_hand_pose=log_map(rotations["right_hand_pose"]).flatten(1),
            jaw_pose=jaw,
            leye_pose=eyes,
            reye_pose=eyes,
            betas=state.betas.expand(t, -1) if state.betas.shape[0] == 1 else state.betas,
            expression=expression,
            transl=state.translation,
            pose2rot=True,
            return_verts=True,
        )
        # DexAvatar/SGNify evaluator convention applies a 180-degree camera-X transform.
        camera_x_180 = state.translation.new_tensor([1.0, -1.0, -1.0])
        return SMPLXOutput(
            vertices=output.vertices * camera_x_180,
            joints=output.joints * camera_x_180,
        )
