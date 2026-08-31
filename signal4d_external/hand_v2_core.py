"""Target-label-free utilities for the external-only V2 hand lane."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)


HAND_REGIONS = {"lhand": (21, 36, 1), "rhand": (36, 51, 2)}
COVERAGE_GRID = (0.10, 0.25, 0.50, 0.75, 1.00)
ALPHA_GRID = (0.25, 0.50, 0.75, 1.00)
SMOOTHING_HALF_WINDOW_SECONDS_GRID = (0.0, 2.0 / 15.0, 4.0 / 15.0)
MIN_HAND_VALID_FRACTION = 0.50
MIN_HAND_RELIABILITY = 0.20


def geodesic_blend(
    initial: torch.Tensor, candidate: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Move from ``initial`` toward ``candidate`` along the SO(3) geodesic."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    relative = candidate.float() @ initial.float().transpose(-1, -2)
    tangent = matrix_to_axis_angle(relative)
    return axis_angle_to_matrix(tangent * float(alpha)) @ initial.float()


def smooth_scores(
    scores: np.ndarray,
    timestamps: np.ndarray,
    half_window_seconds: float,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a centered time-domain moving average within one clip."""
    values = np.asarray(scores, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if times.shape != values.shape or not np.isfinite(times).all():
        raise ValueError("timestamps must be finite and match scores")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    mask = (
        np.ones(len(values), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    if mask.shape != values.shape:
        raise ValueError("eligible must match scores")
    if half_window_seconds < 0:
        raise ValueError("half_window_seconds must be non-negative")
    if half_window_seconds == 0 or len(values) == 0:
        return values.copy()
    result = np.empty_like(values)
    for index, timestamp in enumerate(times):
        start = np.searchsorted(times, timestamp - half_window_seconds, side="left")
        stop = np.searchsorted(times, timestamp + half_window_seconds, side="right")
        local = mask[start:stop]
        result[index] = (
            values[start:stop][local].mean() if local.any() else values[index]
        )
    return result


def smooth_clips(
    scores: Sequence[np.ndarray],
    timestamps: Sequence[np.ndarray],
    half_window_seconds: float,
    eligibilities: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    if len(scores) != len(timestamps):
        raise ValueError("scores and timestamps must contain the same clips")
    masks = (
        [None] * len(scores) if eligibilities is None else list(eligibilities)
    )
    if len(masks) != len(scores):
        raise ValueError("eligibilities must contain the same clips as scores")
    return [
        smooth_scores(value, times, half_window_seconds, mask)
        for value, times, mask in zip(scores, timestamps, masks, strict=True)
    ]


def exact_rank_selection(
    scores: Sequence[np.ndarray],
    coverage: float,
    eligibilities: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Select exactly the top global coverage with stable, index-based ties."""
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must lie in [0, 1]")
    lengths = [len(np.asarray(value)) for value in scores]
    if any(np.asarray(value).ndim != 1 for value in scores):
        raise ValueError("every score sequence must be one-dimensional")
    total = sum(lengths)
    selected = np.zeros(total, dtype=bool)
    eligible = (
        np.ones(total, dtype=bool)
        if eligibilities is None
        else np.concatenate([np.asarray(value, dtype=bool) for value in eligibilities])
    )
    if len(eligible) != total:
        raise ValueError("eligibilities must match score sequence lengths")
    count = int(np.floor(int(eligible.sum()) * coverage + 0.5))
    if count:
        flat = np.concatenate([np.asarray(value, dtype=np.float64) for value in scores])
        # lexsort uses the last key as primary: descending score, then the
        # original global frame index. This makes tied scores reproducible.
        eligible_indices = np.flatnonzero(eligible)
        order = eligible_indices[
            np.lexsort((eligible_indices, -flat[eligible_indices]))
        ]
        selected[order[:count]] = True
    result = []
    start = 0
    for length in lengths:
        result.append(selected[start : start + length].copy())
        start += length
    return result


def selected_count(coverage: float, frame_count: int) -> int:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must lie in [0, 1]")
    return int(np.floor(frame_count * coverage + 0.5))


def hand_eligibility(clip: object, region: str) -> np.ndarray:
    """Target-free hand visibility/reliability safety mask."""
    if region not in HAND_REGIONS:
        raise ValueError(f"Unknown hand region: {region}")
    start, end, _ = HAND_REGIONS[region]
    valid = np.asarray(getattr(clip, "keypoint_valid")[:, start:end], dtype=bool)
    reliability = np.asarray(
        getattr(clip, "u0_reliability")[:, start:end], dtype=np.float32
    )
    return (
        (valid.mean(axis=-1) >= MIN_HAND_VALID_FRACTION)
        & (reliability.mean(axis=-1) >= MIN_HAND_RELIABILITY)
    )
