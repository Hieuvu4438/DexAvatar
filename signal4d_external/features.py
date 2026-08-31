"""Target-statistics-free feature augmentation for external transfer."""

from __future__ import annotations

import torch

from phase2_refiner.data.dataset import (
    KEYPOINT_2D_VALID,
    REPROJECTION_RESIDUAL_2D,
    TOKEN_FEATURE_DIM_WITH_REPROJECTION,
)


REGIONS = ((0, 21), (21, 36), (36, 51))
EXTERNAL_FEATURE_DIM = TOKEN_FEATURE_DIM_WITH_REPROJECTION + 3


def augment_clip_relative_reprojection(
    features: torch.Tensor,
    *,
    minimum_scale: float = 1e-3,
    clip_value: float = 4.0,
) -> torch.Tensor:
    """Append per-clip normalized residual XY and its log median scale.

    The transform uses only observations from the sequence currently being
    inferred.  It never uses a target-domain population statistic or label.
    This makes the representation less sensitive to camera/crop scale while
    retaining the original raw reprojection residual in the first 45 fields.
    """

    if features.ndim != 4:
        raise ValueError(f"Expected [B,T,J,F], got {tuple(features.shape)}")
    if features.shape[-1] != TOKEN_FEATURE_DIM_WITH_REPROJECTION:
        raise ValueError(
            f"Expected {TOKEN_FEATURE_DIM_WITH_REPROJECTION} raw features, "
            f"got {features.shape[-1]}"
        )
    if features.shape[-2] != 51:
        raise ValueError(f"Expected 51 joints, got {features.shape[-2]}")
    if minimum_scale <= 0 or clip_value <= 0:
        raise ValueError("minimum_scale and clip_value must be positive")

    raw = features[..., REPROJECTION_RESIDUAL_2D]
    valid = features[..., KEYPOINT_2D_VALID] > 0.5
    norm = torch.linalg.vector_norm(raw.float(), dim=-1)
    normalized = torch.zeros_like(raw)
    scale_feature = torch.zeros_like(raw[..., :1])
    for batch_index in range(features.shape[0]):
        for start, end in REGIONS:
            region_valid = valid[batch_index, :, start:end]
            values = norm[batch_index, :, start:end][region_valid]
            scale = (
                values.median()
                if values.numel()
                else norm.new_tensor(minimum_scale)
            ).clamp_min(minimum_scale)
            region_raw = raw[batch_index, :, start:end]
            region_normalized = (region_raw / scale.to(region_raw.dtype)).clamp(
                -clip_value, clip_value
            )
            normalized[batch_index, :, start:end] = torch.where(
                region_valid[..., None],
                region_normalized,
                torch.zeros_like(region_normalized),
            )
            scale_feature[batch_index, :, start:end, 0] = torch.log1p(scale).to(
                scale_feature.dtype
            )
    return torch.cat((features, normalized, scale_feature), dim=-1)
