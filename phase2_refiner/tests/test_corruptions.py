import torch

from phase2_refiner.data.corruptions import apply_residual_mixture
from phase2_refiner.data.dataset import ROTATION_6D


def test_t2_residual_mixture_clean_mode_uses_target(monkeypatch):
    features = torch.zeros(2, 4, 51, 43)
    initial = torch.eye(3).expand(2, 4, 51, 3, 3).clone()
    target = initial.clone()
    target[..., 0, 0] = 0.5
    valid = torch.ones(2, 4, 51, dtype=torch.bool)
    frames = torch.ones(2, 4, dtype=torch.bool)
    output_features, output, mask, modes = apply_residual_mixture(
        features,
        initial,
        target,
        frames,
        valid,
        real_fraction=0.0,
        synthetic_fraction=0.0,
        clean_fraction=1.0,
    )
    assert torch.equal(modes, torch.full((2,), 2))
    assert torch.equal(output, target)
    assert not mask.any()
    assert output_features[..., ROTATION_6D].abs().sum() > 0


def test_t2_residual_mixture_rejects_invalid_fractions():
    features = torch.zeros(1, 2, 51, 43)
    matrix = torch.eye(3).expand(1, 2, 51, 3, 3).clone()
    valid = torch.ones(1, 2, 51, dtype=torch.bool)
    frames = torch.ones(1, 2, dtype=torch.bool)
    try:
        apply_residual_mixture(
            features,
            matrix,
            matrix,
            frames,
            valid,
            real_fraction=0.5,
            synthetic_fraction=0.5,
            clean_fraction=0.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mixture did not fail")
