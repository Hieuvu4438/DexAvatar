from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    distribution: np.ndarray


def paired_hierarchical_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    signer_ids: np.ndarray,
    clip_ids: np.ndarray,
    replicates: int = 10000,
    seed: int = 20260819,
) -> BootstrapResult:
    arrays = [np.asarray(value) for value in (candidate, baseline, signer_ids, clip_ids)]
    if len({len(value) for value in arrays}) != 1 or not len(candidate):
        raise ValueError("bootstrap inputs must be non-empty and aligned")
    delta = arrays[0].astype(float) - arrays[1].astype(float)
    rng = np.random.default_rng(seed)
    signers = np.unique(arrays[2])
    distribution = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_values: list[float] = []
        for signer in rng.choice(signers, size=len(signers), replace=True):
            signer_mask = arrays[2] == signer
            signer_clips = np.unique(arrays[3][signer_mask])
            for clip in rng.choice(signer_clips, size=len(signer_clips), replace=True):
                sampled_values.extend(delta[signer_mask & (arrays[3] == clip)].tolist())
        distribution[replicate] = np.mean(sampled_values)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return BootstrapResult(float(delta.mean()), float(low), float(high), distribution)
