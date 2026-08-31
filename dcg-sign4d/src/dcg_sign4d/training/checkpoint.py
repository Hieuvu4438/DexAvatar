"""Immutable, hash-verified model checkpoint artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

Stage = Literal["contact_proposal", "trajectory_diffusion", "ranker"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class CheckpointMetadata:
    stage: Stage
    model_class: str
    step: int
    epoch: int
    seed: int
    config_sha256: str
    manifest_sha256: str
    dependency_commits: dict[str, str]
    metrics: dict[str, float]
    development_only: bool
    asset_sha256: dict[str, str] = field(default_factory=dict)
    schema_version: str = "dcg_checkpoint_v1"

    def validate(self) -> CheckpointMetadata:
        if self.step < 0 or self.epoch < 0:
            raise ValueError("checkpoint step/epoch cannot be negative")
        if not self.model_class:
            raise ValueError("model_class is required")
        for name, value in {
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
        }.items():
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be an exact SHA-256")
        if not self.dependency_commits:
            raise ValueError("dependency commits are required")
        for name, commit in self.dependency_commits.items():
            if not name or not _COMMIT.fullmatch(commit):
                raise ValueError("dependency commits must be named exact 40-hex identities")
        if not all(torch.isfinite(torch.tensor(value)) for value in self.metrics.values()):
            raise ValueError("checkpoint metrics must be finite")
        if not all(name and _SHA256.fullmatch(value) for name, value in self.asset_sha256.items()):
            raise ValueError("checkpoint asset hashes must be named exact SHA-256 values")
        return self


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def save_model_checkpoint(
    destination: str | Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
    *,
    state_scope: Literal["full", "trainable"] = "full",
) -> Path:
    """Atomically write weights plus independently verifiable JSON metadata."""

    metadata.validate()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"immutable checkpoint exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        weights_path = temporary / "weights.pt"
        if state_scope == "full":
            state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        elif state_scope == "trainable":
            state = {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            if not state:
                raise ValueError("trainable-only checkpoint has no trainable parameters")
        else:
            raise ValueError("unknown checkpoint state scope")
        torch.save(state, weights_path)
        total, trainable = _parameter_counts(model)
        payload: dict[str, Any] = {
            **asdict(metadata),
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "weights_sha256": file_sha256(weights_path),
            "state_scope": state_scope,
        }
        payload["metadata_identity_sha256"] = canonical_hash(payload)
        (temporary / "metadata.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "CHECKPOINT_COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_model_checkpoint(
    source: str | Path,
    model: nn.Module,
    *,
    expected_stage: Stage,
    expected_config_sha256: str | None = None,
    allow_development: bool = False,
) -> dict[str, Any]:
    """Verify every identity before strict, weights-only state loading."""

    source = Path(source)
    if not (source / "CHECKPOINT_COMPLETE").is_file():
        raise ValueError("checkpoint has no completion marker")
    metadata_path = source / "metadata.json"
    weights_path = source / "weights.pt"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = payload.pop("metadata_identity_sha256", None)
    if identity != canonical_hash(payload):
        raise ValueError("checkpoint metadata identity mismatch")
    payload["metadata_identity_sha256"] = identity
    if file_sha256(weights_path) != payload.get("weights_sha256"):
        raise ValueError("checkpoint weights hash mismatch")
    if payload.get("stage") != expected_stage:
        raise ValueError("checkpoint stage mismatch")
    if (
        expected_config_sha256 is not None
        and payload.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError("checkpoint config hash mismatch")
    if payload.get("development_only") and not allow_development:
        raise PermissionError("development checkpoint cannot enter a production run")
    expected_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if payload.get("model_class") != expected_class:
        raise ValueError("checkpoint model class mismatch")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("checkpoint is not a tensor-only state dictionary")
    if payload.get("state_scope") == "full":
        model.load_state_dict(state, strict=True)
    elif payload.get("state_scope") == "trainable":
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if parameters.keys() != state.keys():
            raise ValueError("trainable checkpoint parameter topology mismatch")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if parameter.shape != state[name].shape:
                    raise ValueError(f"trainable checkpoint shape mismatch: {name}")
                parameter.copy_(state[name].to(parameter))
    else:
        raise ValueError("unknown checkpoint state scope")
    return payload
