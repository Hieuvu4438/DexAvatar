from __future__ import annotations

import torch

from signal4d.cli.extrapolate import extrapolate_rotations
from signal4d.geometry.so3 import exp_map, geodesic_distance


def test_geodesic_extrapolation_extends_candidate_direction() -> None:
    baseline = torch.eye(3).reshape(1, 3, 3)
    candidate = exp_map(torch.tensor([[0.0, 0.0, 0.2]]))
    result = extrapolate_rotations(baseline, candidate, 2.0)
    expected = exp_map(torch.tensor([[0.0, 0.0, 0.4]]))
    assert float(geodesic_distance(result, expected).max()) < 1e-5
