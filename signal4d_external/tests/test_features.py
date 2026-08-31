from __future__ import annotations

import torch

from signal4d_external.features import augment_clip_relative_reprojection
from signal4d_external.model import ExternalOnlyRefiner


def test_clip_relative_xy_is_scale_invariant() -> None:
    features = torch.zeros(1, 3, 51, 45)
    features[..., 28] = 1.0
    features[..., 43:45] = torch.tensor([0.2, -0.1])
    scaled = features.clone()
    scaled[..., 43:45] *= 7.0
    first = augment_clip_relative_reprojection(features)
    second = augment_clip_relative_reprojection(scaled)
    torch.testing.assert_close(first[..., 45:47], second[..., 45:47])
    assert first.shape[-1] == 48


def test_external_model_forward_contract() -> None:
    model = ExternalOnlyRefiner(
        input_dim=45,
        augmented_input_dim=48,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        max_frames=4,
        predict_uncertainty=False,
        predict_benefit=True,
        use_reprojection_skip=True,
    ).eval()
    features = torch.zeros(1, 4, 51, 45)
    initial = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(1, 4, 51, 3, 3).clone()
    output = model(
        features,
        initial,
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 51, dtype=torch.bool),
    )
    assert output["matrix"].shape == initial.shape
    assert output["benefit_logit"].shape == (1, 4, 3)
    assert torch.isfinite(output["matrix"]).all()
