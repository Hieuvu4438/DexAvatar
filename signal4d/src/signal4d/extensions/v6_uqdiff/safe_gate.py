from __future__ import annotations

import torch
import torch.nn.functional as functional

from ...geometry.so3 import geodesic_distance


def assert_only_open_rotations_changed(
    base: torch.Tensor,
    candidate: torch.Tensor,
    open_joint_indices: tuple[int, ...],
    tolerance_rad: float = 1e-5,
) -> None:
    if base.shape != candidate.shape or base.ndim != 4 or base.shape[-2:] != (3, 3):
        raise ValueError("base and candidate must agree on [T,J,3,3]")
    closed = torch.ones(base.shape[1], dtype=torch.bool, device=base.device)
    closed[list(open_joint_indices)] = False
    drift = geodesic_distance(base[:, closed], candidate[:, closed])
    if float(drift.max()) > tolerance_rad:
        raise ValueError(f"closed SIGNAL4D parameter drifted by {float(drift.max()):.6g} rad")


def safe_acceptance_mask(
    base_objective: torch.Tensor,
    candidate_objective: torch.Tensor,
    max_rotation_delta: torch.Tensor,
    uncertainty_ratio: torch.Tensor,
    *,
    require_objective_improvement: bool,
    minimum_objective_improvement: float = 0.0,
    max_rotation_delta_rad: float,
    max_uncertainty_ratio: float,
    transition_radius: int,
) -> torch.Tensor:
    """GT-free frame gate with conservative temporal rejection dilation."""
    shape = base_objective.shape
    diagnostics = (candidate_objective, max_rotation_delta, uncertainty_ratio)
    if any(value.shape != shape for value in diagnostics):
        raise ValueError("safe-gate diagnostics must share shape [T]")
    if minimum_objective_improvement < 0:
        raise ValueError("minimum objective improvement must be non-negative")
    accept = (max_rotation_delta <= max_rotation_delta_rad) & (
        uncertainty_ratio <= max_uncertainty_ratio
    )
    if require_objective_improvement:
        accept &= candidate_objective + minimum_objective_improvement < base_objective
    if transition_radius > 0 and accept.numel() > 0:
        reject = (~accept).to(dtype=torch.float32)[None, None]
        reject = functional.max_pool1d(
            reject,
            kernel_size=2 * transition_radius + 1,
            stride=1,
            padding=transition_radius,
        )[0, 0].bool()
        accept = ~reject
    return accept


def apply_rotation_gate(
    base: torch.Tensor, candidate: torch.Tensor, accept: torch.Tensor
) -> torch.Tensor:
    if base.shape != candidate.shape or accept.shape != (base.shape[0],):
        raise ValueError("rotation gate expects matching rotations and [T] mask")
    return torch.where(accept[:, None, None, None], candidate, base)
