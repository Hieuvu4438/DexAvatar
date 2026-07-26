"""Safe partial initialization hook for external spatial-prior adapters."""

from __future__ import annotations

from pathlib import Path

import torch


def load_compatible_initialization(
    model: torch.nn.Module, checkpoint: str | Path, minimum_tensors: int = 1
) -> dict:
    """Load compatible tensors and report every decision.

    Dataset/model-specific adapters can translate DPoser-X or another prior into
    this neutral state dict before calling this function. The sole shape adapter
    expands the token input projection when append-only feature channels are
    introduced: learned columns are copied exactly and new columns start at zero.
    Every such adaptation is recorded in the returned provenance.
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
    adapted = []
    projection_key = "token_embedding.input_projection.weight"
    if projection_key in source and projection_key in target:
        source_projection = source[projection_key]
        target_projection = target[projection_key]
        expandable = (
            source_projection.ndim == 2
            and target_projection.ndim == 2
            and source_projection.shape[0] == target_projection.shape[0]
            and source_projection.shape[1] < target_projection.shape[1]
        )
        if expandable:
            expanded = torch.zeros_like(target_projection)
            expanded[:, : source_projection.shape[1]] = source_projection
            compatible[projection_key] = expanded
            adapted.append(
                {
                    "tensor": projection_key,
                    "source_shape": list(source_projection.shape),
                    "target_shape": list(target_projection.shape),
                    "policy": "copy learned prefix; zero-initialize appended features",
                }
            )
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
        "missing_tensor_names": list(missing),
        "unexpected_tensors": len(unexpected),
        "unexpected_tensor_names": list(unexpected),
        "adapted_tensors": adapted,
    }
