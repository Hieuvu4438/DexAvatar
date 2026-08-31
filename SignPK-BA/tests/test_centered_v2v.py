import numpy as np
import torch

from signpk.evaluation.trv2v_audited import translation_aligned_errors
from signpk.losses.centered_vertex import centered_vertex_loss


def test_translation_does_not_change_trv2v():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(30, 3))
    prediction = target + np.array([4.0, -2.0, 8.0])
    np.testing.assert_allclose(translation_aligned_errors(prediction, target), 0, atol=1e-12)


def test_rotation_increases_centered_loss():
    target = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]]])
    rotated = target @ torch.tensor([[0.0, 1, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    assert centered_vertex_loss(rotated, target, [0, 1, 2]) > 0.1

