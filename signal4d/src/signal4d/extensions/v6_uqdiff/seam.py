from __future__ import annotations

import torch

from ...geometry.so3 import geodesic_distance

_LEFT_CHAIN = (0, 3, 6, 9, 13, 16, 18, 20)
_RIGHT_CHAIN = (0, 3, 6, 9, 14, 17, 19, 21)
_LEFT_MCP = (25, 28, 31, 34, 37)
_RIGHT_MCP = (40, 43, 46, 49, 52)


def _chain_product(rotations: torch.Tensor, chain: tuple[int, ...]) -> torch.Tensor:
    result = rotations[:, chain[0]]
    for index in chain[1:]:
        result = result @ rotations[:, index]
    return result


def global_mcp_rotations(rotations: torch.Tensor, side: str) -> torch.Tensor:
    """Compose the SMPL-X arm chain with the five local MCP rotations."""
    if rotations.ndim != 4 or rotations.shape[1:] != (55, 3, 3):
        raise ValueError("canonical rotations must have shape [B,55,3,3]")
    if side == "left":
        wrist, mcp = _chain_product(rotations, _LEFT_CHAIN), _LEFT_MCP
    elif side == "right":
        wrist, mcp = _chain_product(rotations, _RIGHT_CHAIN), _RIGHT_MCP
    else:
        raise ValueError("side must be left or right")
    return wrist[:, None] @ rotations[:, mcp]


def wrist_mcp_seam_loss(
    rotations: torch.Tensor,
    v5_target: torch.Tensor,
    frame_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep refined arm chains compatible with V5/WiLoR global MCP evidence."""
    distances = []
    for side in ("left", "right"):
        distances.append(
            geodesic_distance(
                global_mcp_rotations(rotations, side),
                global_mcp_rotations(v5_target, side),
            )
        )
    squared = torch.cat(distances, dim=1).square()
    if frame_weight is not None:
        if frame_weight.shape != (rotations.shape[0],):
            raise ValueError("frame_weight must have shape [B]")
        squared = squared * frame_weight[:, None]
    return squared.mean()

