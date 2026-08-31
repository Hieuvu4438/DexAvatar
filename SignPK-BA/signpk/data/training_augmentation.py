from __future__ import annotations

from dataclasses import replace

import torch

from signpk.models.explicit_tokens import ExplicitTokenBatch


def _masked_feature(value: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    weights = (~mask).to(value)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    return value * weights


def augment_training_tokens(
    tokens: ExplicitTokenBatch,
    *,
    observation_dropout: float = 0.0,
    feature_mask_probability: float = 0.0,
    token_noise_std: float = 0.0,
) -> ExplicitTokenBatch:
    """Apply window-consistent frozen-observer augmentation.

    Image-space crop/color/blur variants belong in observer extraction. This
    function handles the post-cache augmentations that remain valid in token
    space: per-observer dropout, temporal feature masking, and mild detector
    noise. Base rotations are never changed.
    """

    for name, probability in (
        ("observation_dropout", observation_dropout),
        ("feature_mask_probability", feature_mask_probability),
    ):
        if not 0 <= probability <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if token_noise_std < 0:
        raise ValueError("token_noise_std must be non-negative")
    batch, time = tokens.timestamps.shape
    device = tokens.timestamps.device
    left_drop = torch.rand(batch, 1, device=device) < observation_dropout
    right_drop = torch.rand(batch, 1, device=device) < observation_dropout
    frame_mask = torch.rand(batch, time, device=device) < feature_mask_probability
    left_mask = frame_mask | left_drop
    right_mask = frame_mask | right_drop
    any_hand_mask = left_mask | right_mask

    def noisy(value: torch.Tensor) -> torch.Tensor:
        return value + token_noise_std * torch.randn_like(value) if token_noise_std else value

    result = replace(
        tokens,
        body=_masked_feature(noisy(tokens.body), frame_mask),
        left=_masked_feature(noisy(tokens.left), left_mask),
        right=_masked_feature(noisy(tokens.right), right_mask),
        relation=_masked_feature(noisy(tokens.relation), any_hand_mask),
        left_valid=tokens.left_valid & ~left_mask,
        right_valid=tokens.right_valid & ~right_mask,
        left_observer_feature=_masked_feature(tokens.left_observer_feature, left_mask),
        right_observer_feature=_masked_feature(tokens.right_observer_feature, right_mask),
        left_h4w_feature=_masked_feature(tokens.left_h4w_feature, left_mask),
        right_h4w_feature=_masked_feature(tokens.right_h4w_feature, right_mask),
        body_observer_feature=_masked_feature(tokens.body_observer_feature, frame_mask),
    )
    result.validate()
    return result
