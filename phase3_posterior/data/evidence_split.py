"""Deterministic region-stratified conditioning/evidence masks."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


REGIONS = (slice(0, 21), slice(21, 36), slice(36, 51))


def clip_seed(clip_id: str, fold: int = 0) -> int:
    digest = hashlib.sha256(f"{clip_id}:{fold}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def evidence_mask(
    valid: torch.Tensor,
    clip_id: str,
    fraction: float = 0.2,
    fold: int = 0,
) -> torch.Tensor:
    """Hold out observations independently within body/left/right regions."""
    if valid.ndim != 2 or valid.shape[1] != 51:
        raise ValueError("valid must have shape (T,51)")
    if not 0 <= fraction < 0.5:
        raise ValueError("evidence fraction must be in [0,0.5)")
    result = torch.zeros_like(valid, dtype=torch.bool)
    rng = np.random.default_rng(clip_seed(clip_id, fold))
    for region in REGIONS:
        indices = torch.nonzero(valid[:, region], as_tuple=False)
        if len(indices) < 2:
            continue
        count = min(max(1, int(round(len(indices) * fraction))), len(indices) - 1)
        chosen = rng.choice(len(indices), size=count, replace=False)
        selected = indices[torch.from_numpy(np.asarray(chosen)).long()]
        selected[:, 1] += region.start
        result[selected[:, 0], selected[:, 1]] = True
    return result


def conditioning_mask(valid: torch.Tensor, held_out: torch.Tensor) -> torch.Tensor:
    if valid.shape != held_out.shape:
        raise ValueError("valid and held_out masks must match")
    if torch.any(held_out & ~valid):
        raise ValueError("held-out evidence must be valid")
    return valid & ~held_out
