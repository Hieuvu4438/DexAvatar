from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


def aggregate_subgroups(
    per_frame_errors: Mapping[str, Mapping[str, np.ndarray]],
    attributes: Mapping[str, Mapping[str, str | bool]],
    keys: Iterable[str] = ("handedness", "interaction", "velocity", "disagreement"),
) -> dict[str, dict[str, float]]:
    """Aggregate region errors by declared, non-GT subgroup attributes."""

    buckets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for frame_key, regions in per_frame_errors.items():
        if frame_key not in attributes:
            raise KeyError(f"missing subgroup attributes for {frame_key}")
        for key in keys:
            if key not in attributes[frame_key]:
                continue
            group = f"{key}={attributes[frame_key][key]}"
            for region, values in regions.items():
                buckets[group][region].append(np.asarray(values))
    return {
        group: {
            region: float(np.concatenate(values).mean() * 1000.0)
            for region, values in regions.items()
            if values
        }
        for group, regions in buckets.items()
    }

