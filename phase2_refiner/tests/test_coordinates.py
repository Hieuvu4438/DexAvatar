import torch

from phase2_refiner.geometry.coordinates import invert_transform, transform_points
from phase2_refiner.geometry.palm import palm_normal


def test_coordinate_round_trip() -> None:
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
    points = torch.randn(3, 7, 3)
    transformed = transform_points(points, transform.expand(3, -1, -1))
    recovered = transform_points(
        transformed, invert_transform(transform).expand(3, -1, -1)
    )
    assert torch.allclose(points, recovered, atol=1e-6)


def test_palm_normal_has_consistent_side_sign() -> None:
    hand = torch.zeros(15, 3)
    hand[0] = torch.tensor([1.0, 0.0, 0.0])
    hand[6] = torch.tensor([0.0, 1.0, 0.0])
    left = palm_normal(hand, "left")
    right = palm_normal(hand, "right")
    assert torch.allclose(left, -right)
    assert torch.allclose(left.norm(), torch.tensor(1.0))
