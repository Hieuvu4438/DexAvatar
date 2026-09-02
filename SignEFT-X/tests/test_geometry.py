import numpy as np
import torch

from signeft.hand.refinement import palm_canonical
from signeft.model.kinematics import apply_lie_residual


def synthetic_hand() -> torch.Tensor:
    joints = torch.zeros(1, 21, 3)
    joints[0, 5] = torch.tensor([1.0, 2.0, 0.1])
    joints[0, 9] = torch.tensor([0.0, 2.5, 0.2])
    joints[0, 17] = torch.tensor([-1.0, 2.0, -0.1])
    for index in range(1, 21):
        if not bool(joints[0, index].any()):
            joints[0, index] = torch.tensor([index % 5 - 2.0, index / 8.0, index / 30.0])
    return joints


def test_palm_canonical_removes_similarity_transform() -> None:
    hand = synthetic_hand()
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0],
         [torch.sin(angle), torch.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    transformed = 2.3 * (hand @ rotation.T) + torch.tensor([4.0, -3.0, 2.0])
    first, first_det = palm_canonical(hand)
    second, second_det = palm_canonical(transformed)
    assert torch.allclose(first, second, atol=2e-6)
    assert torch.all(first_det > 0.999)
    assert torch.all(second_det > 0.999)


def test_lie_residual_is_bounded() -> None:
    identity = torch.eye(3).reshape(1, 1, 3, 3)
    delta = torch.tensor([[[1.0, -2.0, 3.0]]])
    radius = np.deg2rad(12.0)
    _, bounded = apply_lie_residual(identity, delta, radius)
    assert float(torch.linalg.vector_norm(bounded, dim=-1).max()) <= radius + 1e-7
