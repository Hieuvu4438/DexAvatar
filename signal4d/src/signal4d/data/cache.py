from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..utils.hashing import sha256_file
from .manifest import ClipManifest


@dataclass
class ObservationBatch:
    frame_ids: torch.Tensor
    joints_3d: torch.Tensor
    valid_3d: torch.Tensor
    features: torch.Tensor
    keypoints_2d: torch.Tensor | None = None
    valid_2d: torch.Tensor | None = None
    rotations: torch.Tensor | None = None
    valid_rot: torch.Tensor | None = None
    camera_K: torch.Tensor | None = None
    image_size: torch.Tensor | None = None

    def validate(self) -> None:
        if self.frame_ids.ndim != 1:
            raise ValueError("frame_ids must have shape [T]")
        t = self.frame_ids.shape[0]
        if (
            self.joints_3d.ndim != 4
            or self.joints_3d.shape[0] != t
            or self.joints_3d.shape[-1] != 3
        ):
            raise ValueError("joints_3d must have shape [T,S,J,3]")
        if self.valid_3d.shape != self.joints_3d.shape[:-1] or self.valid_3d.dtype != torch.bool:
            raise ValueError("valid_3d must be boolean [T,S,J]")
        if self.features.ndim != 4 or self.features.shape[:3] != self.valid_3d.shape:
            raise ValueError("features must have shape [T,S,J,F]")
        for name, value in self.tensors().items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

    def validate_against(self, manifest: ClipManifest) -> None:
        self.validate()
        if self.frame_ids.tolist() != manifest.frame_ids:
            raise ValueError(f"cache frame IDs do not match manifest for {manifest.clip_id}")

    def tensors(self) -> dict[str, torch.Tensor]:
        values = {
            "frame_ids": self.frame_ids,
            "joints_3d": self.joints_3d,
            "valid_3d": self.valid_3d,
            "features": self.features,
            "keypoints_2d": self.keypoints_2d,
            "valid_2d": self.valid_2d,
            "rotations": self.rotations,
            "valid_rot": self.valid_rot,
            "camera_K": self.camera_K,
            "image_size": self.image_size,
        }
        return {
            key: value.detach().contiguous().cpu()
            for key, value in values.items()
            if value is not None
        }

    def save(self, root: str | Path, metadata: dict[str, object]) -> Path:
        self.validate()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "observations.safetensors"
        fd, temp_name = tempfile.mkstemp(dir=root, prefix=".observations.", suffix=".tmp")
        os.close(fd)
        try:
            save_file(self.tensors(), temp_name)
            os.replace(temp_name, target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        meta = dict(metadata)
        meta["artifact_sha256"] = sha256_file(target)
        (root / "metadata.json").write_text(
            json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, root: str | Path) -> tuple[ObservationBatch, dict[str, object]]:
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        tensor_path = root / "observations.safetensors"
        if sha256_file(tensor_path) != metadata.get("artifact_sha256"):
            raise ValueError(f"cache hash mismatch: {tensor_path}")
        values = load_file(tensor_path)
        batch = cls(**values)
        batch.validate()
        return batch, metadata
