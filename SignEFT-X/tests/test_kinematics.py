import torch

from signeft.model.kinematics import (
    apply_lie_residual,
    compensate_wrist,
    so3_exp_map,
    so3_log_map,
    translation_aligned_hand_distance,
)


def test_wrist_compensation_preserves_global_rotation():
    dtype = torch.float64
    parent_base = so3_exp_map(torch.tensor([0.1, -0.2, 0.05], dtype=dtype))
    local_base = so3_exp_map(torch.tensor([-0.05, 0.08, 0.12], dtype=dtype))
    wrist_global = parent_base @ local_base
    parent_new = so3_exp_map(torch.tensor([0.08, 0.02, -0.04], dtype=dtype)) @ parent_base
    local_new = compensate_wrist(parent_new, wrist_global)
    assert torch.linalg.matrix_norm(parent_new @ local_new - wrist_global) < 1e-12


def test_lie_residual_is_bounded_in_forward():
    baseline = torch.eye(3, dtype=torch.float64)
    _, bounded = apply_lie_residual(
        baseline, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), 0.1
    )
    assert torch.allclose(torch.linalg.vector_norm(bounded), torch.tensor(0.1, dtype=torch.float64))


def test_so3_log_exp_round_trip_for_trust_region_rotations():
    value = torch.tensor([[0.2, -0.3, 0.1], [-0.7, 0.1, 0.4]], dtype=torch.float64)
    assert torch.allclose(so3_log_map(so3_exp_map(value)), value, atol=1e-10)


def test_hand_distance_ignores_translation_but_not_rotation():
    hand = torch.randn(2, 100, 3, dtype=torch.float64)
    translated = hand + torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    assert torch.max(translation_aligned_hand_distance(translated, hand)) < 1e-12
    rotation = so3_exp_map(torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64))
    rotated = hand @ rotation.transpose(-1, -2)
    assert torch.min(translation_aligned_hand_distance(rotated, hand)) > 1e-3
