"""Clip-level temporal errors; smoothness alone is never treated as accuracy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _difference(value: FloatArray, fps: float, order: int) -> FloatArray:
    for _ in range(order):
        value = np.diff(value, axis=0) * fps
    return value


def _mean_point_error(source: FloatArray, target: FloatArray) -> float:
    if source.size == 0:
        return float("nan")
    return float(np.linalg.norm(source - target, axis=-1).mean())


def _normalized_power(value: FloatArray) -> FloatArray:
    centered = value - value.mean(axis=0, keepdims=True)
    power = np.square(np.abs(np.fft.rfft(centered, axis=0))).sum(axis=tuple(range(1, value.ndim)))
    return power / max(float(power.sum()), 1e-12)


def temporal_motion_metrics(
    source: FloatArray,
    target: FloatArray,
    *,
    fps: float,
    high_frequency_fraction: float = 0.25,
) -> dict[str, float]:
    """Compare aligned [T,P,3] trajectories in physical time units."""
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("temporal trajectories must have equal [T,P,3] shape")
    if fps <= 0 or not 0 < high_frequency_fraction < 1:
        raise ValueError("invalid fps or high-frequency cutoff")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("temporal trajectories contain NaN/Inf")
    output = {
        "velocity_error_mm_per_s": _mean_point_error(
            _difference(source, fps, 1), _difference(target, fps, 1)
        )
        * 1000,
        "acceleration_error_mm_per_s2": _mean_point_error(
            _difference(source, fps, 2), _difference(target, fps, 2)
        )
        * 1000,
        "jerk_error_mm_per_s3": _mean_point_error(
            _difference(source, fps, 3), _difference(target, fps, 3)
        )
        * 1000,
    }
    source_power, target_power = _normalized_power(source), _normalized_power(target)
    output["spectral_l1_distance"] = float(np.abs(source_power - target_power).sum())
    source_amplitude = float(np.sqrt(np.square(source - source.mean(0)).mean()))
    target_amplitude = float(np.sqrt(np.square(target - target.mean(0)).mean()))
    output["motion_amplitude_ratio"] = source_amplitude / max(target_amplitude, 1e-12)
    start = max(1, int(np.ceil(len(source_power) * high_frequency_fraction)))
    source_high = float(source_power[start:].sum())
    target_high = float(target_power[start:].sum())
    output["high_frequency_energy_ratio"] = source_high / max(target_high, 1e-12)
    return output
