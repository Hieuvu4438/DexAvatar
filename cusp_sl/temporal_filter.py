"""Target-free temporal controls on SO(3) for the CUSP-SL A2 ablation."""

from __future__ import annotations

import torch

from cusp_sl.geometry import (
    axis_angle_to_matrix,
    geodesic_distance,
    matrix_to_axis_angle,
)


def changed_joint_support(
    base: torch.Tensor,
    candidate: torch.Tensor,
    *,
    tolerance_radians: float = 1e-6,
) -> torch.Tensor:
    """Return joints actually modified by A1, independent of any targets."""

    if base.shape != candidate.shape or base.ndim != 4:
        raise ValueError("Base/candidate rotations must share [T,J,3,3]")
    if tolerance_radians < 0:
        raise ValueError("tolerance_radians must be non-negative")
    return (geodesic_distance(base, candidate) > tolerance_radians).any(dim=0)


def centered_tangent_filter(
    rotations: torch.Tensor,
    joint_mask: torch.Tensor,
    *,
    radius: int = 1,
) -> torch.Tensor:
    """Apply a centered triangular moving average in each rotation tangent.

    The rotation at time ``t`` is the tangent reference, so the filter is
    invariant to a common left rotation.  It is a deliberately inexpensive,
    non-learned and target-free A2 control, not a deployable causal smoother.
    """

    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [T,J,3,3], got {tuple(rotations.shape)}")
    if joint_mask.shape != (rotations.shape[1],):
        raise ValueError(
            f"Expected joint mask {(rotations.shape[1],)}, got {tuple(joint_mask.shape)}"
        )
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0 or rotations.shape[0] == 1 or not bool(joint_mask.any()):
        return rotations.clone()

    output = rotations.clone()
    frames = rotations.shape[0]
    for frame in range(frames):
        start = max(0, frame - radius)
        stop = min(frames, frame + radius + 1)
        offsets = torch.arange(start, stop, device=rotations.device) - frame
        weights = (radius + 1 - offsets.abs()).to(rotations.dtype)
        reference = rotations[frame]
        relatives = reference.transpose(-1, -2)[None] @ rotations[start:stop]
        tangent = matrix_to_axis_angle(relatives)
        mean = (tangent * weights[:, None, None]).sum(dim=0) / weights.sum()
        filtered = reference @ axis_angle_to_matrix(mean)
        output[frame, joint_mask] = filtered[joint_mask]
    return output


__all__ = ["centered_tangent_filter", "changed_joint_support"]
