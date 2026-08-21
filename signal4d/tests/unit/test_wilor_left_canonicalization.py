import torch

from signal4d.adapters.sgnify_local import (
    mirror_left_mano_points,
    mirror_left_mano_rotations,
)
from signal4d.geometry.so3 import exp_map, log_map


def test_left_mano_point_mirror_flips_only_x() -> None:
    points = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    expected = torch.tensor([[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0]])
    assert torch.equal(mirror_left_mano_points(points), expected)
    assert torch.equal(points, torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]))


def test_left_mano_rotation_matches_legacy_axis_angle_convention() -> None:
    axis_angle = torch.tensor([[0.2, -0.3, 0.4], [-0.1, 0.25, -0.35]])
    mirrored = mirror_left_mano_rotations(exp_map(axis_angle))
    expected = axis_angle * axis_angle.new_tensor([1.0, -1.0, -1.0])
    assert torch.allclose(log_map(mirrored), expected, atol=1e-6)
    identity = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(mirrored.transpose(-1, -2) @ mirrored, identity, atol=1e-6)
