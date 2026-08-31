import math

import torch

from cusp_sl.geometry import (
    axis_angle_to_matrix, compose_right, gate_from_reliability,
    geodesic_distance, residual_target,
)


def test_right_residual_round_trip():
    base = axis_angle_to_matrix(torch.randn(4, 7, 3) * 0.4)
    delta = torch.randn(4, 7, 3) * 0.2
    target = compose_right(base, delta)
    recovered = residual_target(base, target)
    rebuilt = compose_right(base, recovered)
    assert geodesic_distance(rebuilt, target).max() < 1e-5


def test_identity_gate_is_exact():
    base = axis_angle_to_matrix(torch.randn(2, 3, 51, 3))
    residual = torch.randn(2, 3, 51, 3)
    output = compose_right(base, residual, gate=torch.zeros(2, 3, 51))
    torch.testing.assert_close(output, base, rtol=0, atol=0)


def test_gate_direction():
    q = torch.tensor([0.0, 0.35, 0.55, 0.75, 1.0])
    gate = gate_from_reliability(q, 0.35, 0.75)
    torch.testing.assert_close(gate, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))


def test_gate_dilation_operates_only_along_time():
    probability = torch.ones(1, 5, 2)
    probability[:, 2, 0] = 0.0
    gate = gate_from_reliability(probability, 0.25, 0.75, dilation=1)
    torch.testing.assert_close(gate[0, :, 0], torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(gate[0, :, 1], torch.zeros(5))


def test_near_pi_stays_finite():
    base = torch.eye(3)
    target = axis_angle_to_matrix(torch.tensor([math.pi - 1e-5, 0.0, 0.0]))
    residual = residual_target(base, target)
    assert torch.isfinite(residual).all()
    assert abs(float(residual.norm()) - (math.pi - 1e-5)) < 2e-4
