from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.hashing import sha256_json


@dataclass(frozen=True)
class MeshRegionRegistry:
    model_id: str
    vertex_count: int
    regions: dict[str, tuple[int, ...]]
    schema_version: str = "1.0"

    def validate(self) -> None:
        if self.vertex_count <= 0:
            raise ValueError("vertex_count must be positive")
        for name, indices in self.regions.items():
            if not indices:
                raise ValueError(f"region {name} is empty")
            if min(indices) < 0 or max(indices) >= self.vertex_count:
                raise ValueError(
                    f"region {name} has indices outside vertex count {self.vertex_count}"
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"region {name} contains duplicate indices")

    @property
    def sha256(self) -> str:
        return sha256_json(
            {
                "schema_version": self.schema_version,
                "model_id": self.model_id,
                "vertex_count": self.vertex_count,
                "regions": {key: list(value) for key, value in sorted(self.regions.items())},
            }
        )

    @classmethod
    def from_npy_files(
        cls, model_id: str, vertex_count: int, region_paths: dict[str, str | Path]
    ) -> MeshRegionRegistry:
        registry = cls(
            model_id=model_id,
            vertex_count=vertex_count,
            regions={
                name: tuple(int(value) for value in np.load(path).reshape(-1))
                for name, path in region_paths.items()
            },
        )
        registry.validate()
        return registry

    def write(self, path: str | Path) -> None:
        self.validate()
        payload = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "vertex_count": self.vertex_count,
            "regions": {key: list(value) for key, value in self.regions.items()},
            "sha256": self.sha256,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
