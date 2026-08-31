"""Versioned scientific patch-map asset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dcg_sign4d.utils.hashing import canonical_hash


@dataclass(frozen=True)
class PatchMap:
    version: str
    smplx_model_version: str
    mesh_vertex_count: int
    patches: dict[str, tuple[int, ...]]
    admissible_edges: tuple[tuple[str, str], ...]
    excluded_edges: tuple[tuple[str, str], ...]
    development_only: bool
    content_hash: str
    scientific_status: str | None = None
    source_assets: dict[str, object] | None = None

    @classmethod
    def load(cls, path: str | Path) -> PatchMap:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        canonical = {key: value for key, value in raw.items() if key != "sha256"}
        patches = {name: tuple(indices) for name, indices in raw["patches"].items()}
        edges = tuple(tuple(edge) for edge in raw["admissible_edges"])
        excluded = tuple(tuple(edge) for edge in raw.get("excluded_edges", []))
        result = cls(
            version=raw["patch_map_version"],
            smplx_model_version=raw["smplx_model_version"],
            mesh_vertex_count=int(raw["mesh_vertex_count"]),
            patches=patches,
            admissible_edges=edges,
            excluded_edges=excluded,
            development_only=bool(raw.get("development_only", False)),
            content_hash=canonical_hash(canonical),
            scientific_status=raw.get("scientific_status"),
            source_assets=raw.get("source_assets"),
        )
        result.validate()
        if "sha256" in raw and raw["sha256"] != result.content_hash:
            raise ValueError("patch-map SHA-256 mismatch")
        return result

    def validate(self) -> None:
        if self.mesh_vertex_count <= 0 or not self.patches or not self.admissible_edges:
            raise ValueError("incomplete patch map")
        for name, indices in self.patches.items():
            if not indices or len(indices) != len(set(indices)):
                raise ValueError(f"patch {name!r} is empty or has duplicate vertices")
            if min(indices) < 0 or max(indices) >= self.mesh_vertex_count:
                raise ValueError(f"patch {name!r} has out-of-range vertices")
        normalized: set[tuple[str, str]] = set()
        for edge in (*self.admissible_edges, *self.excluded_edges):
            if len(edge) != 2 or edge[0] not in self.patches or edge[1] not in self.patches:
                raise ValueError(f"invalid patch edge {edge}")
            if edge[0] == edge[1]:
                raise ValueError("self patch edges are not admissible")
            key = tuple(sorted(edge))
            if key in normalized:
                raise ValueError(f"duplicate/conflicting edge {edge}")
            normalized.add(key)
        if not self.development_only and self.scientific_status != "FROZEN":
            raise ValueError("non-development patch maps must declare scientific_status: FROZEN")
