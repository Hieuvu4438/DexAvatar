from __future__ import annotations

import pytest
import torch

from signal4d.extensions.v6_uqdiff.diffusion_prior import (
    canonical_joint_indices,
    euclidean_dposer_loss,
    geodesic_dposer_loss,
    normalized_dimension_mask,
    uncertainty_change_weights,
)
from signal4d.extensions.v6_uqdiff.safe_gate import (
    apply_rotation_gate,
    assert_only_open_rotations_changed,
    safe_acceptance_mask,
)
from signal4d.extensions.v6_uqdiff.seam import wrist_mcp_seam_loss
from signal4d.geometry.so3 import exp_map


def _identity(batch: int = 3) -> torch.Tensor:
    return torch.eye(3).expand(batch, 55, 3, 3).clone()


def test_euclidean_loss_matches_open_dimension_formula() -> None:
    current = torch.zeros(2, 256)
    target = torch.ones_like(current)
    mask = normalized_dimension_mask((17, 18), False, False, device=current.device)
    snr = torch.ones(2, 1)
    expected = 0.5 * torch.sqrt(torch.tensor(2.0)) * int(mask.sum())
    torch.testing.assert_close(euclidean_dposer_loss(current, target, snr, mask), expected)


def test_geodesic_prior_has_finite_gradient() -> None:
    rotations = _identity(2).requires_grad_()
    target = _identity(2)
    target[:, 18] = exp_map(torch.tensor([[0.2, 0.0, 0.0]]).expand(2, -1))
    loss = geodesic_dposer_loss(rotations, target, torch.ones(2, 1), (18,))
    loss.backward()
    assert torch.isfinite(rotations.grad).all()
    assert float(rotations.grad[:, 18].abs().sum()) > 0


def test_uncertainty_and_change_weights_are_monotonic() -> None:
    uncertainty = torch.tensor([[1.0], [2.0], [4.0]])
    change = torch.tensor([0.0, 0.0, 1.0])
    weight = uncertainty_change_weights(
        uncertainty,
        change,
        uncertainty_aware=True,
        change_aware=True,
    )
    assert weight[1] > weight[0]
    assert weight[2] < weight[1]


def test_wrist_mcp_seam_is_zero_at_v5_and_detects_arm_change() -> None:
    base = _identity(2)
    torch.testing.assert_close(wrist_mcp_seam_loss(base, base), torch.tensor(0.0))
    candidate = base.clone()
    candidate[:, 18] = exp_map(torch.tensor([[0.1, 0.0, 0.0]]).expand(2, -1))
    assert wrist_mcp_seam_loss(candidate, base) > 0


def test_safe_gate_dilates_rejections_and_falls_back_exactly() -> None:
    base_objective = torch.ones(7)
    candidate_objective = torch.zeros(7)
    candidate_objective[3] = 2
    accept = safe_acceptance_mask(
        base_objective,
        candidate_objective,
        torch.zeros(7),
        torch.ones(7),
        require_objective_improvement=True,
        max_rotation_delta_rad=0.2,
        max_uncertainty_ratio=1.5,
        transition_radius=1,
    )
    assert accept.tolist() == [True, True, False, False, False, True, True]
    base, candidate = _identity(7), _identity(7)
    candidate[:, 18] = exp_map(torch.tensor([[0.1, 0.0, 0.0]]).expand(7, -1))
    gated = apply_rotation_gate(base, candidate, accept)
    torch.testing.assert_close(gated[3], base[3])
    torch.testing.assert_close(gated[0], candidate[0])


def test_closed_parameter_guard_rejects_drift() -> None:
    base, candidate = _identity(1), _identity(1)
    candidate[:, 10] = exp_map(torch.tensor([[0.1, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="closed SIGNAL4D parameter"):
        assert_only_open_rotations_changed(base, candidate, (18,))
    assert canonical_joint_indices((17,), False, False) == (18,)
