import numpy as np
import torch

from signpccx.geometry.contact_regions import nearest_region
from signpccx.optimization.contact import (
    ContactProposal,
    contact_attraction,
    gated_contact_loss,
    penetration_barrier,
)
from signpccx.optimization.hypotheses import chirality_loss, signed_area_2d, twist_wrist
from signpccx.optimization.post_refine import _per_frame_contact_distance


def test_chirality_rejects_mirror_and_has_gradient():
    observed = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    correct = observed.clone().requires_grad_(True)
    mirrored = observed.clone()
    mirrored[..., 0] *= -1
    assert chirality_loss(correct, observed, (0, 1, 2)) < chirality_loss(mirrored, observed, (0, 1, 2))
    chirality_loss(correct, observed, (0, 1, 2)).backward()
    assert torch.isfinite(correct.grad).all()
    assert signed_area_2d(observed[:, 0], observed[:, 1], observed[:, 2]).item() > 0


def test_wrist_twist_uses_supplied_local_axis():
    wrist = torch.zeros(3)
    axis = torch.tensor([0.0, 0.0, 2.0])
    result = twist_wrist(wrist, axis, 30.0)
    assert torch.allclose(result, torch.tensor([0.0, 0.0, np.pi / 6]), atol=1e-5)


def test_contact_gradient_pulls_toward_target():
    a = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    b = torch.tensor([[[0.02, 0.0, 0.0]]])
    loss = contact_attraction(a, b, target_distance=0.003)
    loss.backward()
    assert torch.isfinite(a.grad).all() and a.grad.norm() > 0
    # Gradient descent subtracts a negative x-gradient, moving A toward B.
    assert a.grad[0, 0, 0] < 0


def test_contact_confidence_gate_and_penetration_gradient():
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]], requires_grad=True)
    regions = {"left_tip": [0], "right_palm": [1]}
    proposals = [ContactProposal("left_tip", "right_palm", 0.69, 0.003)]
    loss, active = gated_contact_loss(vertices, proposals, regions)
    assert active == 0 and loss.item() == 0
    signed = torch.tensor([-0.01, 0.01], requires_grad=True)
    barrier = penetration_barrier(signed)
    barrier.backward()
    assert signed.grad[0] < 0  # gradient descent increases the negative distance
    assert signed.grad[1] == 0


def test_nearest_contact_region_is_deterministic_and_restricted():
    vertices = np.asarray([[0, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float)
    result = nearest_region(vertices, np.asarray([0.9, 0, 0]), np.asarray([0, 2, 3]), k=2)
    assert result.tolist() == [2, 0]


def test_per_frame_contact_distance_does_not_mix_batch_items():
    a = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    b = torch.tensor([[[0.003, 0.0, 0.0]], [[0.020, 0.0, 0.0]]])
    distances = _per_frame_contact_distance(a, b)
    assert torch.allclose(distances, torch.tensor([0.003, 0.020]))
