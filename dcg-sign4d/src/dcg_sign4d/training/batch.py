"""Immutable supervised-window bundles shared by Stage 2 and Stage 3 trainers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch
from dcg_sign4d.utils.hashing import file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
Split = Literal["train", "calibration", "validation", "gold_validation"]


@dataclass(frozen=True)
class SupervisedWindowMetadata:
    split: Split
    manifest_sha256: str
    observation_cache_sha256: str
    patch_map_sha256: str
    contact_label_status: str
    sample_label_status: tuple[str, ...]
    sample_ids: tuple[str, ...]
    edge_names: tuple[tuple[str, str], ...]
    development_only: bool
    schema_version: str = "dcg_supervised_windows_v1"

    def validate(self) -> SupervisedWindowMetadata:
        for name in (
            "manifest_sha256",
            "observation_cache_sha256",
            "patch_map_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be an exact SHA-256")
        if not self.sample_ids:
            raise ValueError("sample IDs must be non-empty")
        normalized_edges = {tuple(sorted(edge)) for edge in self.edge_names}
        if (
            not self.edge_names
            or len(normalized_edges) != len(self.edge_names)
            or any(len(edge) != 2 or edge[0] == edge[1] for edge in self.edge_names)
        ):
            raise ValueError("edge names must be non-empty unique source/target pairs")
        allowed = {"gold", "accepted_pseudo", "synthetic_fixture"}
        if len(self.sample_label_status) != len(self.sample_ids) or any(
            value not in allowed for value in self.sample_label_status
        ):
            raise ValueError("sample label statuses must align with sample IDs")
        allowed_summary = allowed | {"mixed"}
        if self.contact_label_status not in allowed_summary:
            raise ValueError("unknown contact label status")
        if self.contact_label_status != "mixed" and any(
            value != self.contact_label_status for value in self.sample_label_status
        ):
            raise ValueError("summary and per-sample contact label statuses disagree")
        if not self.development_only and "synthetic_fixture" in self.sample_label_status:
            raise ValueError("synthetic labels cannot enter production training")
        return self


@dataclass(frozen=True)
class SupervisedWindowBatch:
    trajectory: TrajectoryState
    observations: ObservationBatch
    geometry_features: Tensor
    graph: ContactGraphBatch
    duration_frames: Tensor
    supervision_mask: Tensor | None
    metadata: SupervisedWindowMetadata

    def validate(self) -> SupervisedWindowBatch:
        self.trajectory.validate()
        self.observations.validate()
        self.graph.validate()
        self.metadata.validate()
        batch, time = self.trajectory.valid_mask.shape
        if self.observations.frame_valid.shape != (batch, time):
            raise ValueError("observation/trajectory batch-time mismatch")
        if self.graph.event_state.shape[:2] != (batch, time):
            raise ValueError("contact/trajectory batch-time mismatch")
        edges = self.graph.event_state.shape[2]
        if len(self.metadata.edge_names) != edges:
            raise ValueError("metadata edge-name count does not match graph topology")
        if self.geometry_features.shape != (batch, time, edges, 5):
            raise ValueError("geometry features must be [B,T,E,5]")
        if self.duration_frames.shape != (batch, time, edges):
            raise ValueError("duration frames must be [B,T,E]")
        if self.duration_frames.dtype != torch.long or bool((self.duration_frames < 1).any()):
            raise ValueError("duration frames must be positive long values")
        encoded, _ = StateCodec().encode(self.trajectory)
        if self.supervision_mask is not None and (
            self.supervision_mask.shape != encoded.shape
            or self.supervision_mask.dtype != torch.bool
        ):
            raise ValueError("supervision mask must be bool [B,T,D]")
        if len(self.metadata.sample_ids) != batch:
            raise ValueError("sample ID count must equal batch size")
        if not torch.isfinite(self.geometry_features).all():
            raise ValueError("geometry features contain NaN/Inf")
        return self

    def select(self, indices: Tensor) -> SupervisedWindowBatch:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("batch indices must be long [N]")

        def selected(value: Tensor | None) -> Tensor | None:
            return value.index_select(0, indices.to(value.device)) if value is not None else None

        trajectory = replace(
            self.trajectory,
            **{
                name: selected(getattr(self.trajectory, name))
                for name in self.trajectory.__dataclass_fields__
                if isinstance(getattr(self.trajectory, name), Tensor)
            },
        )
        observations = replace(
            self.observations,
            **{
                name: selected(getattr(self.observations, name))
                for name in self.observations.__dataclass_fields__
                if isinstance(getattr(self.observations, name), Tensor)
            },
            metadata=tuple(
                self.observations.metadata[index]
                for index in indices.cpu().tolist()
                if self.observations.metadata
            ),
        )
        graph = replace(
            self.graph,
            **{
                name: selected(getattr(self.graph, name))
                for name in self.graph.__dataclass_fields__
            },
        )
        metadata = replace(
            self.metadata,
            sample_ids=tuple(self.metadata.sample_ids[index] for index in indices.cpu().tolist()),
            sample_label_status=tuple(
                self.metadata.sample_label_status[index] for index in indices.cpu().tolist()
            ),
        )
        return SupervisedWindowBatch(
            trajectory,
            observations,
            selected(self.geometry_features),
            graph,
            selected(self.duration_frames),
            selected(self.supervision_mask),
            metadata,
        ).validate()

    def to(self, device: torch.device | str) -> SupervisedWindowBatch:
        indices = torch.arange(self.trajectory.valid_mask.shape[0])
        batch = self.select(indices)

        def moved(value: Tensor | None) -> Tensor | None:
            return value.to(device) if value is not None else None

        return replace(
            batch,
            trajectory=replace(
                batch.trajectory,
                **{
                    name: moved(getattr(batch.trajectory, name))
                    for name in batch.trajectory.__dataclass_fields__
                    if isinstance(getattr(batch.trajectory, name), Tensor)
                },
            ),
            observations=replace(
                batch.observations,
                **{
                    name: moved(getattr(batch.observations, name))
                    for name in batch.observations.__dataclass_fields__
                    if isinstance(getattr(batch.observations, name), Tensor)
                },
            ),
            geometry_features=moved(batch.geometry_features),
            graph=replace(
                batch.graph,
                **{
                    name: moved(getattr(batch.graph, name))
                    for name in batch.graph.__dataclass_fields__
                },
            ),
            duration_frames=moved(batch.duration_frames),
            supervision_mask=moved(batch.supervision_mask),
        ).validate()


def _arrays(batch: SupervisedWindowBatch) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for prefix, value in (
        ("trajectory", batch.trajectory),
        ("observation", batch.observations),
        ("graph", batch.graph),
    ):
        for name in value.__dataclass_fields__:
            tensor = getattr(value, name)
            if isinstance(tensor, Tensor):
                result[f"{prefix}__{name}"] = tensor.detach().cpu().numpy()
    result["geometry_features"] = batch.geometry_features.detach().cpu().numpy()
    result["duration_frames"] = batch.duration_frames.detach().cpu().numpy()
    if batch.supervision_mask is not None:
        result["supervision_mask"] = batch.supervision_mask.detach().cpu().numpy()
    return result


def save_supervised_windows(
    destination: str | Path,
    batch: SupervisedWindowBatch,
) -> Path:
    batch.validate()
    if len(set(batch.metadata.sample_ids)) != len(batch.metadata.sample_ids):
        raise ValueError("persisted supervised-bundle sample IDs must be unique")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"immutable supervised bundle exists: {destination}")
    destination.mkdir(parents=True)
    incomplete = destination / ".bundle_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    data_path = destination / "windows.npz"
    np.savez_compressed(data_path, **_arrays(batch))
    metadata: dict[str, Any] = {
        **batch.metadata.__dict__,
        "sample_ids": list(batch.metadata.sample_ids),
        "sample_label_status": list(batch.metadata.sample_label_status),
        "edge_names": [list(edge) for edge in batch.metadata.edge_names],
        "windows_sha256": file_sha256(data_path),
        "batch_size": batch.trajectory.valid_mask.shape[0],
        "window_length": batch.trajectory.valid_mask.shape[1],
        "trajectory_dimension": StateCodec().encode(batch.trajectory)[0].shape[-1],
        "edge_count": batch.graph.event_state.shape[2],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, destination / "BUNDLE_COMPLETE")
    return destination


def load_supervised_windows(
    source: str | Path,
    *,
    expected_split: Split,
    allow_development: bool = False,
) -> SupervisedWindowBatch:
    source = Path(source)
    if not (source / "BUNDLE_COMPLETE").is_file():
        raise ValueError("supervised bundle has no completion marker")
    metadata_payload = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata_payload.get("schema_version") != "dcg_supervised_windows_v1":
        raise ValueError("unknown supervised bundle schema")
    if metadata_payload.get("split") != expected_split:
        raise ValueError("supervised bundle split mismatch")
    if metadata_payload.get("development_only") and not allow_development:
        raise PermissionError("development bundle cannot enter production training")
    data_path = source / "windows.npz"
    if file_sha256(data_path) != metadata_payload.get("windows_sha256"):
        raise ValueError("supervised bundle hash mismatch")
    with np.load(data_path, allow_pickle=False) as arrays:
        values = {name: torch.from_numpy(arrays[name]) for name in arrays.files}
    required_arrays = {
        "trajectory__root_rot6d",
        "trajectory__root_translation",
        "trajectory__root_velocity",
        "trajectory__body_rot6d",
        "trajectory__left_hand_rot6d",
        "trajectory__right_hand_rot6d",
        "trajectory__beta",
        "trajectory__valid_mask",
        "observation__keypoints_2d",
        "observation__keypoint_reliability",
        "observation__keypoint_valid",
        "observation__frame_valid",
        "graph__event_state",
        "graph__event_probability",
        "graph__edge_valid",
        "graph__uncertain_mask",
        "graph__segment_id",
        "graph__segment_duration",
        "geometry_features",
        "duration_frames",
    }
    optional_arrays = {
        "trajectory__face_state",
        "observation__part_masks",
        "observation__mask_reliability",
        "observation__tracks_2d",
        "observation__track_reliability",
        "observation__depth_order",
        "observation__depth_reliability",
        "supervision_mask",
    }
    if not required_arrays <= values.keys() or not values.keys() <= (
        required_arrays | optional_arrays
    ):
        raise ValueError("supervised bundle tensor schema mismatch")

    def tensor(prefix: str, name: str, *, boolean: bool = False) -> Tensor | None:
        value = values.get(f"{prefix}__{name}")
        return value.bool() if value is not None and boolean else value

    trajectory = TrajectoryState(
        root_rot6d=tensor("trajectory", "root_rot6d"),
        root_translation=tensor("trajectory", "root_translation"),
        root_velocity=tensor("trajectory", "root_velocity"),
        body_rot6d=tensor("trajectory", "body_rot6d"),
        left_hand_rot6d=tensor("trajectory", "left_hand_rot6d"),
        right_hand_rot6d=tensor("trajectory", "right_hand_rot6d"),
        beta=tensor("trajectory", "beta"),
        valid_mask=tensor("trajectory", "valid_mask", boolean=True),
        face_state=tensor("trajectory", "face_state"),
    )
    observations = ObservationBatch(
        keypoints_2d=tensor("observation", "keypoints_2d"),
        keypoint_reliability=tensor("observation", "keypoint_reliability"),
        keypoint_valid=tensor("observation", "keypoint_valid", boolean=True),
        frame_valid=tensor("observation", "frame_valid", boolean=True),
        part_masks=tensor("observation", "part_masks"),
        mask_reliability=tensor("observation", "mask_reliability"),
        tracks_2d=tensor("observation", "tracks_2d"),
        track_reliability=tensor("observation", "track_reliability"),
        depth_order=tensor("observation", "depth_order"),
        depth_reliability=tensor("observation", "depth_reliability"),
        metadata=tuple({"sample_id": item} for item in metadata_payload["sample_ids"]),
    )
    graph = ContactGraphBatch(
        event_state=tensor("graph", "event_state").long(),
        event_probability=tensor("graph", "event_probability"),
        edge_valid=tensor("graph", "edge_valid", boolean=True),
        uncertain_mask=tensor("graph", "uncertain_mask", boolean=True),
        segment_id=tensor("graph", "segment_id").long(),
        segment_duration=tensor("graph", "segment_duration"),
    )
    metadata = SupervisedWindowMetadata(
        split=metadata_payload["split"],
        manifest_sha256=metadata_payload["manifest_sha256"],
        observation_cache_sha256=metadata_payload["observation_cache_sha256"],
        patch_map_sha256=metadata_payload["patch_map_sha256"],
        contact_label_status=metadata_payload["contact_label_status"],
        sample_label_status=tuple(metadata_payload["sample_label_status"]),
        sample_ids=tuple(metadata_payload["sample_ids"]),
        edge_names=tuple(tuple(edge) for edge in metadata_payload["edge_names"]),
        development_only=bool(metadata_payload["development_only"]),
    )
    return SupervisedWindowBatch(
        trajectory=trajectory,
        observations=observations,
        geometry_features=values["geometry_features"],
        graph=graph,
        duration_frames=values["duration_frames"].long(),
        supervision_mask=values.get("supervision_mask", None),
        metadata=metadata,
    ).validate()
