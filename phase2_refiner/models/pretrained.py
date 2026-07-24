"""Safe partial initialization hook for external spatial-prior adapters."""

from __future__ import annotations

from pathlib import Path

import torch


def load_compatible_initialization(
    model: torch.nn.Module, checkpoint: str | Path, minimum_tensors: int = 1
) -> dict:
    """Load only identically named/shaped tensors and report every decision.

    Dataset/model-specific adapters can translate DPoser-X or another prior into
    this neutral state dict before calling this function. Silent shape coercion
    is deliberately forbidden.
    """
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    source = (
        payload.get("ema_model")
        or payload.get("model")
        or payload.get("state_dict", payload)
    )
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    if len(compatible) < minimum_tensors:
        raise ValueError(
            f"Only {len(compatible)} compatible tensors in spatial initialization; "
            f"required {minimum_tensors}"
        )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "loaded_tensors": len(compatible),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }
