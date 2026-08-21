import math

import torch

from signal4d.geometry.alignment import mean_point_error_mm, procrustes_align, translation_align
from signal4d.geometry.handedness import mirror_points_x
from signal4d.geometry.so3 import (
    exp_map,
    geodesic_distance,
    log_map,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    slerp,
)


def test_exp_log_round_trip() -> None:
    torch.manual_seed(1)
    vectors = torch.randn(128, 3, dtype=torch.float64)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True) * torch.rand(128, 1) * (math.pi - 0.1)
    recovered = exp_map(log_map(exp_map(vectors)))
    assert torch.allclose(recovered, exp_map(vectors), atol=2e-6)


def test_geodesic_invariance() -> None:
    first = exp_map(torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64))
    second = exp_map(torch.tensor([-0.4, 0.2, 0.1], dtype=torch.float64))
    common = exp_map(torch.tensor([0.1, 0.5, -0.2], dtype=torch.float64))
    expected = geodesic_distance(first, second)
    assert torch.allclose(expected, geodesic_distance(common @ first, common @ second), atol=1e-7)
    assert torch.allclose(expected, geodesic_distance(first @ common, second @ common), atol=1e-7)


def test_rotation_6d_is_proper() -> None:
    rotation = rotation_6d_to_matrix(torch.randn(32, 6, dtype=torch.float64))
    identity = torch.eye(3, dtype=torch.float64).expand(32, 3, 3)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-7)
    assert torch.allclose(torch.det(rotation), torch.ones(32, dtype=torch.float64), atol=1e-7)
    assert torch.allclose(
        rotation_6d_to_matrix(matrix_to_rotation_6d(rotation)), rotation, atol=1e-7
    )


def test_alignment_firewall() -> None:
    torch.manual_seed(2)
    target = torch.randn(3, 20, 3, dtype=torch.float64)
    translation = target + torch.tensor([2.0, -1.0, 0.4])
    assert mean_point_error_mm(translation_align(translation, target), target).max() < 1e-9
    rotation = exp_map(torch.tensor([0.2, 0.4, -0.3], dtype=torch.float64))
    rotated = target @ rotation
    assert mean_point_error_mm(translation_align(rotated, target), target).mean() > 1
    assert mean_point_error_mm(procrustes_align(rotated, target), target).max() < 1e-6
    scaled = target * 1.7
    assert mean_point_error_mm(translation_align(scaled, target), target).mean() > 1
    assert mean_point_error_mm(procrustes_align(scaled, target), target).max() < 1e-6


def test_slerp_and_double_mirror() -> None:
    first = torch.eye(3, dtype=torch.float64)
    second = exp_map(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(slerp(first, second, 0.0), first, atol=1e-8)
    assert torch.allclose(slerp(first, second, 1.0), second, atol=1e-8)
    assert torch.allclose(
        geodesic_distance(first, slerp(first, second, 0.5)),
        torch.tensor(0.5, dtype=torch.float64),
        atol=1e-7,
    )
    points = torch.tensor([[1.0, 2.0], [4.0, 3.0]])
    assert torch.equal(mirror_points_x(mirror_points_x(points, 10), 10), points)
