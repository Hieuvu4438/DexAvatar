from __future__ import annotations

import torch

from ...geometry.so3 import geodesic_distance


def normalized_dimension_mask(
    open_body_indices: tuple[int, ...],
    optimize_left_hand: bool,
    optimize_right_hand: bool,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map open SMPL-X groups into DPoser-X's 256-D normalized vector."""
    mask = torch.zeros(256, dtype=torch.bool, device=device)
    for index in open_body_indices:
        mask[index * 3 : (index + 1) * 3] = True
    if optimize_left_hand:
        mask[63:108] = True
    if optimize_right_hand:
        mask[108:153] = True
    return mask


def canonical_joint_indices(
    open_body_indices: tuple[int, ...],
    optimize_left_hand: bool,
    optimize_right_hand: bool,
) -> tuple[int, ...]:
    values = [index + 1 for index in open_body_indices]
    if optimize_left_hand:
        values.extend(range(25, 40))
    if optimize_right_hand:
        values.extend(range(40, 55))
    return tuple(values)


def uncertainty_change_weights(
    uncertainty: torch.Tensor,
    change_probability: torch.Tensor,
    *,
    uncertainty_aware: bool,
    change_aware: bool,
    change_gamma: float = 2.0,
    minimum_change_weight: float = 0.1,
) -> torch.Tensor:
    """Increase prior trust under expert uncertainty and preserve fast transitions."""
    if uncertainty.ndim != 2 or change_probability.shape != (uncertainty.shape[0],):
        raise ValueError("uncertainty [T,J] and change_probability [T] are required")
    weight = torch.ones_like(uncertainty)
    if uncertainty_aware:
        reference = uncertainty.median(dim=0).values.clamp_min(1e-6)
        weight = (uncertainty / reference).clamp(0.25, 4.0)
    if change_aware:
        motion_weight = minimum_change_weight + (1 - minimum_change_weight) * (
            1 - change_probability
        ).pow(change_gamma)
        weight = weight * motion_weight[:, None]
    return weight


def euclidean_dposer_loss(
    normalized: torch.Tensor,
    target: torch.Tensor,
    snr: torch.Tensor,
    dimension_mask: torch.Tensor,
    frame_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Published DPoser-X MSE/SNR objective restricted to open dimensions."""
    if normalized.shape != target.shape or normalized.ndim != 2:
        raise ValueError("normalized DPoser-X tensors must agree on [B,D]")
    if dimension_mask.shape != (normalized.shape[1],):
        raise ValueError("dimension mask does not match normalized pose dimension")
    if not bool(dimension_mask.any()):
        raise ValueError("at least one DPoser-X dimension must be open")
    if snr.shape not in {(normalized.shape[0],), (normalized.shape[0], 1)}:
        raise ValueError("SNR must have shape [B] or [B,1]")
    weight = 0.5 * torch.sqrt(1 + snr.reshape(-1, 1).square())
    if frame_weight is not None:
        if frame_weight.shape != (normalized.shape[0],):
            raise ValueError("frame_weight must have shape [B]")
        weight = weight * frame_weight[:, None]
    squared = (normalized - target).square()
    return (weight * squared[:, dimension_mask]).sum() / normalized.shape[0]


def geodesic_dposer_loss(
    rotations: torch.Tensor,
    target: torch.Tensor,
    snr: torch.Tensor,
    joint_indices: tuple[int, ...],
    joint_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pull a frozen DPoser-X target back to the SO(3) product manifold."""
    if rotations.shape != target.shape or rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotation tensors must agree on [B,J,3,3]")
    if not joint_indices:
        raise ValueError("at least one canonical rotation must be open")
    distance = geodesic_distance(rotations[:, joint_indices], target[:, joint_indices])
    weight = 0.5 * torch.sqrt(1 + snr.reshape(-1, 1).square())
    if joint_weight is not None:
        if joint_weight.shape != rotations.shape[:2]:
            raise ValueError("joint_weight must have shape [B,J]")
        weight = weight * joint_weight[:, joint_indices]
    return (weight * distance.square()).sum() / rotations.shape[0]

