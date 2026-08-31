from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .rotations import project_to_so3


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def make_palm_frame(joints: Tensor, side: str, eps: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    """Construct semantic palm bases from canonical 21-joint MANO ordering."""

    if joints.shape[-2:] != (21, 3):
        raise ValueError(f"expected MANO joints [...,21,3], got {tuple(joints.shape)}")
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    wrist = joints[..., WRIST, :]
    x_raw = joints[..., INDEX_MCP, :] - joints[..., PINKY_MCP, :]
    y_raw = joints[..., MIDDLE_MCP, :] - wrist
    cross = torch.cross(x_raw, y_raw, dim=-1)
    scale = torch.linalg.vector_norm(x_raw, dim=-1) * torch.linalg.vector_norm(y_raw, dim=-1)
    conditioning = torch.linalg.vector_norm(cross, dim=-1) / scale.clamp_min(eps)
    valid = conditioning > 1e-3
    x = F.normalize(x_raw, dim=-1, eps=eps)
    z = F.normalize(cross, dim=-1, eps=eps)
    y = F.normalize(torch.cross(z, x, dim=-1), dim=-1, eps=eps)
    if side == "left":
        x, z = -x, -z
    basis = project_to_so3(torch.stack([x, y, z], dim=-1))
    identity = torch.eye(3, dtype=joints.dtype, device=joints.device).expand_as(basis)
    basis = torch.where(valid[..., None, None], basis, identity)
    return basis, wrist, valid


def fill_invalid_palm_frames(frames: Tensor, valid: Tensor) -> Tensor:
    """Replicate the nearest previous/next valid frame without hiding validity."""

    if frames.ndim != 3 or frames.shape[-2:] != (3, 3) or valid.shape != frames.shape[:1]:
        raise ValueError("expected frames [T,3,3] and valid [T]")
    result = frames.clone()
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_ids.numel() == 0:
        return result
    for index in range(len(result)):
        if not valid[index]:
            nearest = valid_ids[torch.argmin(torch.abs(valid_ids - index))]
            result[index] = result[nearest]
    return result

