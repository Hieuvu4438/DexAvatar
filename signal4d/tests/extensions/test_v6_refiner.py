from __future__ import annotations

import torch

from signal4d.extensions.v6_uqdiff.refiner import compose_tangent_update
from signal4d.extensions.v6_uqdiff.retraction import stable_exp_map
from signal4d.geometry.so3 import geodesic_distance


def test_stable_exp_map_has_finite_nonzero_gradient_at_zero() -> None:
    tangent = torch.zeros(2, 3, requires_grad=True)
    target = stable_exp_map(torch.tensor([[0.2, 0.0, 0.0]]).expand(2, -1))
    loss = geodesic_distance(stable_exp_map(tangent), target).square().mean()
    loss.backward()
    assert torch.isfinite(tangent.grad).all()
    assert float(tangent.grad.abs().sum()) > 0


def test_tangent_composition_keeps_closed_rotations_bitwise() -> None:
    base = torch.eye(3).expand(2, 55, 3, 3).clone()
    delta = torch.zeros(2, 2, 3)
    delta[:, 0, 0] = 0.1
    result = compose_tangent_update(base, delta, (17, 18), None, None)
    closed = torch.ones(55, dtype=torch.bool)
    closed[[18, 19]] = False
    assert torch.equal(result[:, closed], base[:, closed])
    assert not torch.equal(result[:, 18], base[:, 18])
