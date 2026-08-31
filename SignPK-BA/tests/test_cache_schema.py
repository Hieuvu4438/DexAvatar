from pathlib import Path

import torch

from signpk.data.cache_schema import (
    BodyObservation,
    CACHE_SCHEMA_VERSION,
    DualObserverBundle,
    HandObservation,
    ObserverBundle,
    load_dual_observer_bundle,
    load_observer_bundle,
    save_dual_observer_bundle,
    save_observer_bundle,
)


def _hand(frames: int) -> HandObservation:
    identity = torch.eye(3)
    return HandObservation(
        pose_rotmat=identity.expand(frames, 16, 3, 3).clone(),
        shape=torch.zeros(frames, 10),
        vertices_local=torch.zeros(frames, 778, 3),
        joints_local=torch.zeros(frames, 21, 3),
        palm_rotmat=identity.expand(frames, 3, 3).clone(),
        wrist_world_rel=torch.zeros(frames, 3),
        bbox_xyxy=torch.ones(frames, 4),
        keypoints2d=torch.zeros(frames, 21, 2),
        keypoint_confidence=torch.ones(frames, 21),
        confidence=torch.ones(frames),
        valid=torch.ones(frames, dtype=torch.bool),
    )


def test_observer_bundle_roundtrip(tmp_path: Path):
    frames = 2
    identity = torch.eye(3)
    body = BodyObservation(
        root_rotmat=identity.expand(frames, 3, 3).clone(),
        body_rotmat=identity.expand(frames, 21, 3, 3).clone(),
        shape=torch.zeros(frames, 10),
        vertices=torch.zeros(frames, 10475, 3),
        joints3d=torch.zeros(frames, 55, 3),
        keypoints2d=torch.zeros(frames, 42, 2),
        keypoint_confidence=torch.ones(frames, 42),
        translation=torch.zeros(frames, 3),
        focal_length=torch.ones(frames, 2),
        principal_point=torch.zeros(frames, 2),
    )
    bundle = ObserverBundle(
        body=body,
        left=_hand(frames),
        right=_hand(frames),
        root_rel=torch.zeros(frames, 3),
        frame_ids=torch.tensor([1, 3]),
        timestamps=torch.tensor([0.1, 0.2]),
        metadata={"schema_version": CACHE_SCHEMA_VERSION},
    )
    path = tmp_path / "cache.pt"
    save_observer_bundle(bundle, path)
    loaded = load_observer_bundle(path)
    assert loaded.frame_ids.tolist() == [1, 3]
    assert loaded.metadata["schema_version"] == CACHE_SCHEMA_VERSION

    dual = DualObserverBundle(
        body=body,
        h4w_left=_hand(frames),
        h4w_right=_hand(frames),
        omni_left=_hand(frames),
        omni_right=_hand(frames),
        root_rel=torch.zeros(frames, 3),
        frame_ids=torch.tensor([1, 3]),
        timestamps=torch.tensor([0.1, 0.2]),
        metadata={"schema_version": CACHE_SCHEMA_VERSION},
    )
    dual_path = tmp_path / "dual.pt"
    save_dual_observer_bundle(dual, dual_path)
    restored = load_dual_observer_bundle(dual_path)
    assert restored.frame_ids.tolist() == [1, 3]
    assert restored.omni_left.vertices_local.shape == (frames, 778, 3)
