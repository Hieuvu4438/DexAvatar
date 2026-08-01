import torch

from phase2_refiner.data.corruptions import (
    apply_burst_corruption,
    apply_residual_mixture,
)
from phase2_refiner.data.dataset import (
    REPROJECTION_RESIDUAL_2D,
    ROTATION_6D,
    TOKEN_FEATURE_DIM_WITH_REPROJECTION,
)


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


def test_t2_non_real_modes_do_not_reuse_frozen_initializer_reprojection():
    features = torch.zeros(1, 4, 51, TOKEN_FEATURE_DIM_WITH_REPROJECTION)
    features[..., REPROJECTION_RESIDUAL_2D] = 0.25
    initial = torch.eye(3).expand(1, 4, 51, 3, 3).clone()
    target = initial.clone()
    valid = torch.ones(1, 4, 51, dtype=torch.bool)
    frames = torch.ones(1, 4, dtype=torch.bool)

    clean_features, _, _, _ = apply_residual_mixture(
        features,
        initial,
        target,
        frames,
        valid,
        real_fraction=0.0,
        synthetic_fraction=0.0,
        clean_fraction=1.0,
    )
    real_features, _, _, _ = apply_residual_mixture(
        features,
        initial,
        target,
        frames,
        valid,
        real_fraction=1.0,
        synthetic_fraction=0.0,
        clean_fraction=0.0,
    )

    assert torch.count_nonzero(
        clean_features[..., REPROJECTION_RESIDUAL_2D]
    ) == 0
    assert torch.equal(
        real_features[..., REPROJECTION_RESIDUAL_2D],
        features[..., REPROJECTION_RESIDUAL_2D],
    )


def test_burst_corruption_zeros_stale_reprojection_for_affected_joints():
    features = torch.ones(1, 4, 51, TOKEN_FEATURE_DIM_WITH_REPROJECTION)
    matrix = torch.eye(3).expand(1, 4, 51, 3, 3).clone()
    frames = torch.ones(1, 4, dtype=torch.bool)
    corrupted, _, mask = apply_burst_corruption(
        features,
        matrix,
        frames,
        probability=1.0,
        min_duration=4,
        max_duration=4,
        modes=["left_hand"],
    )
    assert mask[:, :, 21:36].all()
    assert torch.count_nonzero(
        corrupted[:, :, 21:36, REPROJECTION_RESIDUAL_2D]
    ) == 0
    assert torch.equal(
        corrupted[:, :, 36:51, REPROJECTION_RESIDUAL_2D],
        features[:, :, 36:51, REPROJECTION_RESIDUAL_2D],
    )
