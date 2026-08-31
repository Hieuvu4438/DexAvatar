import pytest
import torch

from cusp_sl.geometry import axis_angle_to_matrix, geodesic_distance
from cusp_sl.temporal_filter import centered_tangent_filter, changed_joint_support


def test_filter_preserves_constant_sequence_and_unmasked_joints():
    constant = axis_angle_to_matrix(torch.randn(4, 3) * 0.4)
    rotations = constant[None].expand(7, -1, -1, -1).clone()
    rotations[:, 3] = axis_angle_to_matrix(torch.randn(7, 3) * 0.5)
    before_unmasked = rotations[:, 3].clone()
    mask = torch.tensor([True, True, True, False])
    filtered = centered_tangent_filter(rotations, mask, radius=1)
    torch.testing.assert_close(filtered[:, :3], rotations[:, :3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(filtered[:, 3], before_unmasked)


def test_filter_reduces_single_frame_rotation_spike_and_stays_in_so3():
    axis_angle = torch.zeros(5, 2, 3)
    axis_angle[2, 0, 0] = 0.9
    rotations = axis_angle_to_matrix(axis_angle)
    filtered = centered_tangent_filter(
        rotations, torch.tensor([True, False]), radius=1
    )
    identity = torch.eye(3)
    before = geodesic_distance(rotations[2, 0], identity)
    after = geodesic_distance(filtered[2, 0], identity)
    assert after < before
    torch.testing.assert_close(
        filtered.transpose(-1, -2) @ filtered,
        torch.eye(3).expand_as(filtered),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(torch.linalg.det(filtered), torch.ones(5, 2))


def test_filter_validates_radius_and_shapes():
    rotations = torch.eye(3).expand(2, 3, 3).clone()
    with pytest.raises(ValueError, match="Expected"):
        centered_tangent_filter(rotations, torch.ones(2, dtype=torch.bool))
    rotations = torch.eye(3).expand(2, 2, 3, 3).clone()
    with pytest.raises(ValueError, match="non-negative"):
        centered_tangent_filter(rotations, torch.ones(2, dtype=torch.bool), radius=-1)


def test_changed_support_is_target_free_and_exact_outside_a1_edits():
    base = torch.eye(3).expand(5, 4, 3, 3).clone()
    candidate = base.clone()
    candidate[2, 3] = axis_angle_to_matrix(torch.tensor([0.2, 0.0, 0.0]))
    support = changed_joint_support(base, candidate)
    assert support.tolist() == [False, False, False, True]
    filtered = centered_tangent_filter(candidate, support, radius=1)
    torch.testing.assert_close(filtered[:, :3], base[:, :3])
