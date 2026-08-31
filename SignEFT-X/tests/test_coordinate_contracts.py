from pathlib import Path

import numpy as np
import torch

from signeft.data.manifest import read_manifest
from signeft.optim.core import full_to_heatmap, project


ROOT = Path(__file__).parents[1]


def test_exported_pose_crop_round_trip_is_subpixel():
    record = read_manifest(ROOT / "manifests" / "splits" / "engineering12.jsonl")[0]
    path = ROOT / "observations" / "sapiens_pose_v1" / record.sign_id / f"{record.source_frame_id:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        full = torch.as_tensor(archive["coords_full"])[None]
        transform = torch.as_tensor(archive["crop_to_full"])[None]
    low = full_to_heatmap(full, transform)
    homogeneous = torch.cat((low, torch.ones_like(low[..., :1])), dim=-1)
    restored = torch.einsum("bij,bnj->bni", transform, homogeneous)
    restored = restored[..., :2] / restored[..., 2:3]
    assert torch.max(torch.abs(restored - full)) < 0.25


def test_torch_camera_projection_matches_numpy_baseline_convention():
    record = read_manifest(ROOT / "manifests" / "splits" / "engineering12.jsonl")[0]
    with np.load(record.a3f_state_path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"][:128], dtype=np.float32)
        camera = np.asarray(archive["K"], dtype=np.float32)
    assert np.max(vertices[:, 2]) < 0
    expected_h = vertices @ camera.T
    expected = expected_h[:, :2] / expected_h[:, 2:3]
    actual = project(torch.as_tensor(vertices)[None], torch.as_tensor(camera)[None])[0].numpy()
    assert np.max(np.abs(actual - expected)) < 1e-4


def test_nlf_cache_is_meter_scale_and_evaluator_handed():
    record = read_manifest(ROOT / "manifests" / "splits" / "engineering12.jsonl")[0]
    path = ROOT / "observations" / "nlf_v1" / record.sign_id / f"{record.source_frame_id:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(archive["joint_names"].tolist())
        joints = np.asarray(archive["joints3d"])
        assert str(archive["unit"]) == "meter"
        assert str(archive["coord_frame"]) == "evaluator_camera_centered"
    index = {name: i for i, name in enumerate(names)}
    lengths = []
    for parent, child in (
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ):
        lengths.append(np.linalg.norm(joints[:, index[child]] - joints[:, index[parent]], axis=-1))
    assert 0.1 < float(np.median(lengths)) < 0.6
    assert np.mean(joints[:, index["left_shoulder"], 0]) > np.mean(joints[:, index["right_shoulder"], 0])
