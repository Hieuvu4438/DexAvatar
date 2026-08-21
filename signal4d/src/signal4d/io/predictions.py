from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..utils.hashing import sha256_file


@dataclass
class PredictionArtifact:
    frame_ids: torch.Tensor
    joints_3d: torch.Tensor
    rotations: torch.Tensor | None
    translation: torch.Tensor
    vertices: torch.Tensor | None
    risk_score: torch.Tensor
    abstain: torch.Tensor
    uncertainty: torch.Tensor
    contact_probability: torch.Tensor | None = None
    contacts: torch.Tensor | None = None

    def validate(self) -> None:
        t = self.frame_ids.numel()
        if self.frame_ids.ndim != 1 or len(set(self.frame_ids.tolist())) != t:
            raise ValueError("prediction frame_ids must be unique [T]")
        if (
            self.joints_3d.ndim != 3
            or self.joints_3d.shape[0] != t
            or self.joints_3d.shape[-1] != 3
        ):
            raise ValueError("joints_3d must have shape [T,J,3]")
        joint_count = self.joints_3d.shape[1]
        if self.rotations is not None and self.rotations.shape != (t, joint_count, 3, 3):
            raise ValueError("rotations must have shape [T,J,3,3]")
        if self.translation.shape != (t, 3):
            raise ValueError("translation must have shape [T,3]")
        if self.vertices is not None and (
            self.vertices.ndim != 3 or self.vertices.shape[0] != t or self.vertices.shape[-1] != 3
        ):
            raise ValueError("vertices must have shape [T,V,3]")
        if self.risk_score.shape != (t, 3) or self.abstain.shape != (t, 3):
            raise ValueError("risk_score and abstain must have shape [T,3]")
        if self.uncertainty.shape != (t, joint_count):
            raise ValueError("uncertainty must have shape [T,J]")
        if (self.contact_probability is None) != (self.contacts is None):
            raise ValueError("contact probability and hard contacts must be stored together")
        if self.contact_probability is not None and (
            self.contact_probability.ndim != 2
            or self.contact_probability.shape[0] != t
            or self.contacts is None
            or self.contacts.shape != self.contact_probability.shape
        ):
            raise ValueError("contact probability and contacts must agree on [T,C]")
        if self.abstain.dtype != torch.bool:
            raise ValueError("abstain must be boolean")
        if self.contacts is not None and self.contacts.dtype != torch.bool:
            raise ValueError("contacts must be boolean")
        for name, value in self.tensors().items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"prediction tensor {name} contains non-finite values")

    def tensors(self) -> dict[str, torch.Tensor]:
        values = {
            "frame_ids": self.frame_ids,
            "joints_3d": self.joints_3d,
            "rotations": self.rotations,
            "translation": self.translation,
            "vertices": self.vertices,
            "risk_score": self.risk_score,
            "abstain": self.abstain,
            "uncertainty": self.uncertainty,
            "contact_probability": self.contact_probability,
            "contacts": self.contacts,
        }
        return {
            key: value.detach().contiguous().cpu()
            for key, value in values.items()
            if value is not None
        }

    def save(self, root: str | Path, metadata: dict[str, object]) -> None:
        self.validate()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "prediction.safetensors"
        fd, temp_name = tempfile.mkstemp(dir=root, prefix=".prediction.", suffix=".tmp")
        os.close(fd)
        try:
            save_file(self.tensors(), temp_name)
            os.replace(temp_name, target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        meta = dict(metadata)
        meta["artifact_sha256"] = sha256_file(target)
        meta["frame_ids"] = self.frame_ids.tolist()
        (root / "metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, root: str | Path) -> tuple[PredictionArtifact, dict[str, object]]:
        root = Path(root)
        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        target = root / "prediction.safetensors"
        if sha256_file(target) != meta.get("artifact_sha256"):
            raise ValueError(f"prediction hash mismatch: {target}")
        values = load_file(target)
        prediction = cls(
            frame_ids=values["frame_ids"],
            joints_3d=values["joints_3d"],
            rotations=values.get("rotations"),
            translation=values["translation"],
            vertices=values.get("vertices"),
            risk_score=values["risk_score"],
            abstain=values["abstain"],
            uncertainty=values["uncertainty"],
            contact_probability=values.get("contact_probability"),
            contacts=values.get("contacts"),
        )
        prediction.validate()
        return prediction, meta
