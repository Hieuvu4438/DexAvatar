from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


CACHE_SCHEMA_VERSION = "signpk-observer-v1"


@dataclass
class HandObservation:
    pose_rotmat: Tensor
    shape: Tensor
    vertices_local: Tensor
    joints_local: Tensor
    palm_rotmat: Tensor
    wrist_world_rel: Tensor
    bbox_xyxy: Tensor
    keypoints2d: Tensor
    keypoint_confidence: Tensor
    confidence: Tensor
    valid: Tensor
    temporal_token: Tensor | None = None
    vertices_world_rel: Tensor | None = None
    joints_world_rel: Tensor | None = None
    reprojection_error: Tensor | None = None
    padding_ratio: Tensor | None = None

    def validate(self) -> None:
        length = self.pose_rotmat.shape[0]
        required = {
            "pose_rotmat": (length, 16, 3, 3),
            "shape": (length, 10),
            "vertices_local": (length, 778, 3),
            "joints_local": (length, 21, 3),
            "palm_rotmat": (length, 3, 3),
            "wrist_world_rel": (length, 3),
            "bbox_xyxy": (length, 4),
            "keypoints2d": (length, 21, 2),
            "keypoint_confidence": (length, 21),
            "confidence": (length,),
            "valid": (length,),
        }
        for name, shape in required.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN/Inf")
        if self.valid.dtype != torch.bool:
            raise TypeError("hand valid mask must be bool")
        for name in (
            "temporal_token",
            "vertices_world_rel",
            "joints_world_rel",
            "reprojection_error",
            "padding_ratio",
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                if tensor.shape[0] != length:
                    raise ValueError(f"{name} length does not match the hand observation")
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"{name} contains NaN/Inf")


@dataclass
class BodyObservation:
    root_rotmat: Tensor
    body_rotmat: Tensor
    shape: Tensor
    vertices: Tensor
    joints3d: Tensor
    keypoints2d: Tensor
    keypoint_confidence: Tensor
    translation: Tensor
    focal_length: Tensor
    principal_point: Tensor
    body_features: Tensor | None = None

    def validate(self) -> None:
        length = self.root_rotmat.shape[0]
        expected_prefixes = {
            "root_rotmat": (length, 3, 3),
            "body_rotmat": (length, 21, 3, 3),
            "shape": (length, 10),
            "vertices": (length, 10475, 3),
            "translation": (length, 3),
            "focal_length": (length, 2),
            "principal_point": (length, 2),
        }
        for name, shape in expected_prefixes.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN/Inf")
        if self.joints3d.shape[0] != length or self.joints3d.shape[-1] != 3:
            raise ValueError("invalid joints3d shape")
        if self.keypoints2d.shape[:1] != (length,) or self.keypoints2d.shape[-1] != 2:
            raise ValueError("invalid keypoints2d shape")
        if self.keypoint_confidence.shape != self.keypoints2d.shape[:-1]:
            raise ValueError("2D confidence shape does not match keypoints")
        if self.body_features is not None:
            if self.body_features.shape[0] != length:
                raise ValueError("body feature length does not match the body observation")
            if not torch.isfinite(self.body_features).all():
                raise ValueError("body features contain NaN/Inf")


@dataclass
class ObserverBundle:
    body: BodyObservation
    left: HandObservation
    right: HandObservation
    root_rel: Tensor
    frame_ids: Tensor
    timestamps: Tensor
    metadata: dict[str, Any]

    def validate(self) -> None:
        self.body.validate()
        self.left.validate()
        self.right.validate()
        length = self.body.root_rotmat.shape[0]
        if tuple(self.root_rel.shape) != (length, 3):
            raise ValueError("root_rel must have shape [T,3]")
        if tuple(self.frame_ids.shape) != (length,) or tuple(self.timestamps.shape) != (length,):
            raise ValueError("frame IDs/timestamps must have shape [T]")
        if len(set(self.frame_ids.tolist())) != length:
            raise ValueError("frame IDs are not unique")
        if self.metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported observer cache schema")


@dataclass
class DualObserverBundle:
    """Canonical frozen-expert sequence used to prepare PKC training windows."""

    body: BodyObservation
    h4w_left: HandObservation
    h4w_right: HandObservation
    omni_left: HandObservation
    omni_right: HandObservation
    root_rel: Tensor
    frame_ids: Tensor
    timestamps: Tensor
    metadata: dict[str, Any]

    def validate(self) -> None:
        self.body.validate()
        for hand in (
            self.h4w_left,
            self.h4w_right,
            self.omni_left,
            self.omni_right,
        ):
            hand.validate()
        length = self.body.root_rotmat.shape[0]
        if any(
            hand.pose_rotmat.shape[0] != length
            for hand in (
                self.h4w_left,
                self.h4w_right,
                self.omni_left,
                self.omni_right,
            )
        ):
            raise ValueError("dual observer streams have different lengths")
        if self.root_rel.shape != (length, 3):
            raise ValueError("dual observer root_rel must have shape [T,3]")
        if self.frame_ids.shape != (length,) or self.timestamps.shape != (length,):
            raise ValueError("dual observer frame IDs/timestamps must have shape [T]")
        if len(set(self.frame_ids.tolist())) != length:
            raise ValueError("dual observer frame IDs are not unique")
        if self.metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported dual observer cache schema")


@dataclass
class CouplerPrediction:
    root_rotmat: Tensor
    upper_rotmat: Tensor
    left_hand_rotmat: Tensor
    right_hand_rotmat: Tensor
    angular_velocity: Tensor
    wrist_velocity: Tensor
    log_variance: dict[str, Tensor]
    phase_gate: Tensor
    interaction_gate: Tensor

    @property
    def pose_rotmat(self) -> Tensor:
        return torch.cat([self.upper_rotmat, self.left_hand_rotmat, self.right_hand_rotmat], dim=1)


def _dataclass_to_payload(prefix: str, value: Any, payload: dict[str, Any]) -> None:
    for field in fields(value):
        item = getattr(value, field.name)
        key = f"{prefix}.{field.name}"
        if isinstance(item, Tensor):
            payload[key] = item.detach().cpu()
        elif item is None:
            payload[key] = None


def save_observer_bundle(bundle: ObserverBundle, path: str | Path) -> None:
    bundle.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "root_rel": bundle.root_rel.detach().cpu(),
        "frame_ids": bundle.frame_ids.detach().cpu(),
        "timestamps": bundle.timestamps.detach().cpu(),
        "metadata_json": json.dumps(bundle.metadata, sort_keys=True),
    }
    _dataclass_to_payload("body", bundle.body, payload)
    _dataclass_to_payload("left", bundle.left, payload)
    _dataclass_to_payload("right", bundle.right, payload)
    torch.save(payload, path)


def load_observer_bundle(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> ObserverBundle:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)

    def construct(cls, prefix: str):
        return cls(**{field.name: payload.get(f"{prefix}.{field.name}") for field in fields(cls)})

    bundle = ObserverBundle(
        body=construct(BodyObservation, "body"),
        left=construct(HandObservation, "left"),
        right=construct(HandObservation, "right"),
        root_rel=payload["root_rel"],
        frame_ids=payload["frame_ids"],
        timestamps=payload["timestamps"],
        metadata=json.loads(payload["metadata_json"]),
    )
    bundle.validate()
    return bundle


def save_dual_observer_bundle(bundle: DualObserverBundle, path: str | Path) -> None:
    bundle.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "root_rel": bundle.root_rel.detach().cpu(),
        "frame_ids": bundle.frame_ids.detach().cpu(),
        "timestamps": bundle.timestamps.detach().cpu(),
        "metadata_json": json.dumps(bundle.metadata, sort_keys=True),
    }
    for name in ("body", "h4w_left", "h4w_right", "omni_left", "omni_right"):
        _dataclass_to_payload(name, getattr(bundle, name), payload)
    torch.save(payload, path)


def load_dual_observer_bundle(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> DualObserverBundle:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported dual observer cache schema")

    def construct(cls, prefix: str):
        return cls(**{field.name: payload.get(f"{prefix}.{field.name}") for field in fields(cls)})

    bundle = DualObserverBundle(
        body=construct(BodyObservation, "body"),
        h4w_left=construct(HandObservation, "h4w_left"),
        h4w_right=construct(HandObservation, "h4w_right"),
        omni_left=construct(HandObservation, "omni_left"),
        omni_right=construct(HandObservation, "omni_right"),
        root_rel=payload["root_rel"],
        frame_ids=payload["frame_ids"],
        timestamps=payload["timestamps"],
        metadata=json.loads(payload["metadata_json"]),
    )
    bundle.validate()
    return bundle
