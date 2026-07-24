import math

import torch

from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    compose_residual,
    geodesic_distance,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


def test_rotation_round_trips() -> None:
    torch.manual_seed(7)
    axis = torch.randn(256, 3)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.rand(256, 1) * (math.pi - 1e-3)
    matrix = axis_angle_to_matrix(axis * angle)
    from_6d = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))
    from_axis = axis_angle_to_matrix(matrix_to_axis_angle(matrix))
    from_quaternion = quaternion_to_matrix(matrix_to_quaternion(matrix))
    assert torch.allclose(matrix, from_6d, atol=1e-5)
    assert torch.allclose(matrix, from_axis, atol=1e-5)
    assert torch.allclose(matrix, from_quaternion, atol=1e-5)


def test_zero_residual_is_exact_identity() -> None:
    initial = axis_angle_to_matrix(torch.randn(4, 51, 3) * 0.2)
    output = compose_residual(initial, torch.zeros(4, 51, 3), gate=torch.ones(4, 51, 1))
    assert torch.equal(initial, output)
    assert torch.equal(geodesic_distance(initial, output), torch.zeros(4, 51))


def test_identity_geodesic_has_finite_zero_gradient() -> None:
    axis_angle = torch.zeros(2, 3, requires_grad=True)
    matrix = axis_angle_to_matrix(axis_angle)
    loss = geodesic_distance(matrix, matrix.detach()).sum()
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(axis_angle.grad).all()


def test_matrix_axis_angle_round_trip_has_finite_decoder_gradient() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, -0.2, 0.3], [1.5, 0.1, -0.4]],
        requires_grad=True,
    )
    matrix = axis_angle_to_matrix(source)
    recovered = matrix_to_axis_angle(matrix)
    loss = recovered.square().sum()
    loss.backward()
    assert torch.isfinite(recovered).all()
    assert torch.isfinite(source.grad).all()


def test_residual_bound() -> None:
    initial = torch.eye(3).expand(2, 3, 3)
    output = compose_residual(initial, torch.full((2, 3), 10.0), max_angle=0.25)
    distance = geodesic_distance(output, initial)
    assert torch.all(distance <= 0.250001)
