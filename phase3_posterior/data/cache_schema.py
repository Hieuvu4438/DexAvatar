"""Append-only Phase 3 index and relation-sidecar contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    NUM_RELATION_NODES,
)


SCHEMA_VERSION = 1
RELATION_SCHEMA_VERSION = 2
FORBIDDEN_SOURCE_PARTS = ("data/smplx_gt", "data/evaluation_from_author")


def reject_forbidden_path(path: str | Path) -> None:
    normalized = Path(path).resolve().as_posix()
    for forbidden in FORBIDDEN_SOURCE_PARTS:
        if f"/{forbidden}/" in f"/{normalized}/" or normalized.endswith(forbidden):
            raise ValueError(
                f"Forbidden SGNify/evaluation source in Phase 3 data: {path}"
            )


@dataclass(frozen=True)
class Phase3IndexEntry:
    clip_id: str
    clip_path: str
    source: str
    domain: str
    split: str
    source_group: str
    signer: str
    target_weight: float
    license_id: str
    clip_sha256: str
    relation_path: str = ""
    relation_sha256: str = ""

    def validate(self) -> None:
        reject_forbidden_path(self.clip_path)
        if self.relation_path:
            reject_forbidden_path(self.relation_path)
        if not self.clip_id or not self.source or not self.split:
            raise ValueError("clip_id, source, and split are required")
        if not 0 <= self.target_weight <= 1:
            raise ValueError("target_weight must be in [0,1]")
        if len(self.clip_sha256) != 64:
            raise ValueError("clip_sha256 must be a SHA-256 digest")
        if self.relation_path and len(self.relation_sha256) != 64:
            raise ValueError("relation_sha256 must accompany relation_path")


@dataclass
class RelationSidecar:
    clip_id: str
    node_positions: np.ndarray
    node_valid: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_valid: np.ndarray
    contact_target: np.ndarray
    contact_valid: np.ndarray
    persistence_target: np.ndarray
    depth_target: np.ndarray
    target_edge_features: np.ndarray | None = None
    metadata_json: str = "{}"

    def validate(self) -> None:
        t = self.node_positions.shape[0]
        if self.node_positions.shape != (t, NUM_RELATION_NODES, 3):
            raise ValueError(
                f"node_positions has invalid shape {self.node_positions.shape}"
            )
        if self.node_valid.shape != (t, NUM_RELATION_NODES):
            raise ValueError(f"node_valid has invalid shape {self.node_valid.shape}")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2,E)")
        edges = self.edge_index.shape[1]
        expected = {
            "edge_features": (t, edges, EDGE_FEATURE_DIM),
            "edge_valid": (t, edges),
            "contact_target": (t, edges),
            "contact_valid": (t, edges),
            "persistence_target": (t, edges),
            "depth_target": (t, edges),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        if self.target_edge_features is not None and self.target_edge_features.shape != (
            t,
            edges,
            EDGE_FEATURE_DIM,
        ):
            raise ValueError(
                "target_edge_features: expected "
                f"{(t, edges, EDGE_FEATURE_DIM)}, got {self.target_edge_features.shape}"
            )
        if t == 0:
            raise ValueError("relation sidecar cannot be empty")
        if (
            self.edge_index.min(initial=0) < 0
            or self.edge_index.max(initial=0) >= NUM_RELATION_NODES
        ):
            raise ValueError("edge_index contains an invalid node")
        finite_names = ["node_positions", "edge_features"]
        if self.target_edge_features is not None:
            finite_names.append("target_edge_features")
        for name in finite_names:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if not np.isin(self.depth_target, (0, 1, 2)).all():
            raise ValueError("depth_target must use classes 0/1/2")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode a mapping")


def save_relation_sidecar(path: str | Path, sidecar: RelationSidecar) -> None:
    sidecar.validate()
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite relation sidecar: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        payload = asdict(sidecar)
        if payload["target_edge_features"] is None:
            payload.pop("target_edge_features")
            version = 1
        else:
            version = RELATION_SCHEMA_VERSION
        np.savez_compressed(handle, **payload, schema_version=np.asarray(version))
    temporary.replace(target)


def load_relation_sidecar(path: str | Path) -> RelationSidecar:
    with np.load(path, allow_pickle=False) as data:
        version = int(data["schema_version"])
        if version not in {1, RELATION_SCHEMA_VERSION}:
            raise ValueError(f"Unsupported Phase 3 relation schema: {version}")
        sidecar = RelationSidecar(
            clip_id=str(data["clip_id"]),
            node_positions=data["node_positions"],
            node_valid=data["node_valid"],
            edge_index=data["edge_index"],
            edge_features=data["edge_features"],
            edge_valid=data["edge_valid"],
            contact_target=data["contact_target"],
            contact_valid=data["contact_valid"],
            persistence_target=data["persistence_target"],
            depth_target=data["depth_target"],
            target_edge_features=(
                data["target_edge_features"]
                if "target_edge_features" in data.files
                else None
            ),
            metadata_json=str(data["metadata_json"]),
        )
    sidecar.validate()
    return sidecar


def load_index(path: str | Path) -> list[Phase3IndexEntry]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Invalid Phase 3 index: {source}")
    entries = [Phase3IndexEntry(**item) for item in payload.get("clips", [])]
    for entry in entries:
        entry.validate()
    if not entries:
        raise ValueError(f"Phase 3 index is empty: {source}")
    return entries
