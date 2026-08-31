"""External-only SIGNAL4D temporal residual architecture."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from phase2_refiner.models import WholeSequenceRefiner

from .features import EXTERNAL_FEATURE_DIM, augment_clip_relative_reprojection


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalOnlyRefiner(nn.Module):
    """Whole-sequence SO(3) refiner with clip-relative domain normalization."""

    raw_input_dim = 45

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        config = dict(model_config)
        raw_input_dim = int(config.pop("input_dim", self.raw_input_dim))
        augmented_input_dim = int(
            config.pop("augmented_input_dim", EXTERNAL_FEATURE_DIM)
        )
        config.pop("external_init_checkpoint", None)
        config.pop("external_init_sha256", None)
        if raw_input_dim != self.raw_input_dim:
            raise ValueError(f"External lane requires raw input_dim={self.raw_input_dim}")
        if augmented_input_dim != EXTERNAL_FEATURE_DIM:
            raise ValueError(
                f"External lane requires augmented_input_dim={EXTERNAL_FEATURE_DIM}"
            )
        self.backbone = WholeSequenceRefiner(
            input_dim=augmented_input_dim,
            **config,
        )

    @property
    def max_frames(self) -> int:
        return self.backbone.max_frames

    @property
    def max_angles(self) -> torch.Tensor:
        return self.backbone.max_angles

    @property
    def predict_uncertainty(self) -> bool:
        return self.backbone.predict_uncertainty

    def forward(
        self,
        features: torch.Tensor,
        initial_matrix: torch.Tensor,
        frame_valid: torch.Tensor,
        refine_mask: torch.Tensor,
        initial_joint_position: torch.Tensor | None = None,
        uncertainty_offset: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        augmented = augment_clip_relative_reprojection(features)
        return self.backbone(
            augmented,
            initial_matrix,
            frame_valid,
            refine_mask,
            initial_joint_position,
            uncertainty_offset,
        )


def load_external_initialization(
    model: ExternalOnlyRefiner,
    checkpoint: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load an external checkpoint, adapting only the three appended inputs."""

    path = Path(checkpoint).resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"External initialization hash mismatch: expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source = payload.get("ema_model") or payload["model"]
    wrapper_prefix_stripped = False
    if source and all(name.startswith("backbone.") for name in source):
        # External checkpoints store the same backbone below the append-only
        # ExternalOnlyRefiner wrapper.  Accept them for exact-target fine-tuning
        # without weakening shape or hash checks.
        source = {
            name.removeprefix("backbone."): value for name, value in source.items()
        }
        wrapper_prefix_stripped = True
    target = model.backbone.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    zero_initialized_inputs = 0
    for name, value in source.items():
        if name == "max_angles":
            # The new lane deliberately uses tighter, preregistered trust bounds.
            skipped.append(name)
            continue
        if name not in target:
            skipped.append(name)
            continue
        if value.shape == target[name].shape:
            adapted[name] = value
            continue
        if name == "token_embedding.input_projection.weight":
            if value.shape[0] != target[name].shape[0] or value.shape[1] != 45:
                raise ValueError(f"Unsupported input projection shape: {value.shape}")
            expanded = torch.zeros_like(target[name])
            expanded[:, : value.shape[1]] = value
            adapted[name] = expanded
            zero_initialized_inputs = int(target[name].shape[1] - value.shape[1])
            continue
        skipped.append(name)
    missing, unexpected = model.backbone.load_state_dict(adapted, strict=False)
    permitted_missing = {"benefit_head.weight", "benefit_head.bias", "max_angles"}
    invalid_missing = sorted(set(missing) - permitted_missing)
    if invalid_missing or unexpected:
        raise ValueError(
            f"Incompatible external initialization: missing={invalid_missing}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        "checkpoint": str(path),
        "sha256": actual_sha256,
        "loaded_tensors": len(adapted),
        "skipped_source_tensors": sorted(skipped),
        "zero_initialized_inputs": zero_initialized_inputs,
        "zero_initialized_benefit_head": bool(
            {"benefit_head.weight", "benefit_head.bias"} & set(missing)
        ),
        "wrapper_prefix_stripped": wrapper_prefix_stripped,
    }


def model_from_config(config: dict[str, Any], *, initialize: bool) -> ExternalOnlyRefiner:
    model_config = dict(config.get("model", {}))
    model = ExternalOnlyRefiner(**model_config)
    if initialize:
        checkpoint = model_config.get("external_init_checkpoint")
        expected = model_config.get("external_init_sha256")
        if not checkpoint or not expected:
            raise ValueError(
                "Training requires model.external_init_checkpoint and "
                "model.external_init_sha256"
            )
        model.initialization_report = load_external_initialization(
            model, checkpoint, expected
        )
    return model
