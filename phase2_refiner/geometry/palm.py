"""Hand landmark conventions and stable palm geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F


HAND_JOINTS = 15
FINGER_CHAINS = {
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "pinky": (6, 7, 8),
    "ring": (9, 10, 11),
    "thumb": (12, 13, 14),
}
FINGERTIP_INDICES = (2, 5, 8, 11, 14)
MCP_INDICES = (0, 3, 6, 9, 12)


def palm_center(hand_joints: torch.Tensor) -> torch.Tensor:
    if hand_joints.shape[-2:] != (HAND_JOINTS, 3):
        raise ValueError(f"Expected (...,15,3), got {tuple(hand_joints.shape)}")
    return hand_joints[..., list(MCP_INDICES), :].mean(dim=-2)


def palm_normal(hand_joints: torch.Tensor, side: str) -> torch.Tensor:
    """Return a consistently oriented palm normal for SMPL-X hand ordering."""
    center = palm_center(hand_joints)
    index_mcp = hand_joints[..., FINGER_CHAINS["index"][0], :] - center
    pinky_mcp = hand_joints[..., FINGER_CHAINS["pinky"][0], :] - center
    normal = torch.cross(index_mcp, pinky_mcp, dim=-1)
    if side == "right":
        normal = -normal
    elif side != "left":
        raise ValueError("side must be 'left' or 'right'")
    return F.normalize(normal, dim=-1, eps=1e-8)


def fingertips(hand_joints: torch.Tensor) -> torch.Tensor:
    if hand_joints.shape[-2:] != (HAND_JOINTS, 3):
        raise ValueError(f"Expected (...,15,3), got {tuple(hand_joints.shape)}")
    return hand_joints[..., list(FINGERTIP_INDICES), :]
