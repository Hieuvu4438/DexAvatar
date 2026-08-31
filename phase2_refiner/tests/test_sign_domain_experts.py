import json
import numpy as np
import torch

from phase2_refiner.data.build_wilor_frame_manifest import build
from phase2_refiner.data.build_sign_domain_cache import (
    _expert_disagreement,
    _load_hamer_outputs,
    _wilor_hands,
)
from phase2_refiner.data.extract_sign_domain_smplerx import _take_by_dataset
from phase2_refiner.data.shard_wilor_frame_manifest import shard
from phase2_refiner.data.materialize_dexavatar_source_inputs import (
    _camera_from_bbox,
    _normalize_hamer_entry,
)


def test_take_by_dataset_locks_each_provider_limit() -> None:
    entries = [
        {"dataset": "SOKE", "clip_id": f"p{i}"} for i in range(3)
    ] + [{"dataset": "SignAvatars", "clip_id": f"w{i}"} for i in range(4)]
    selected = _take_by_dataset(entries, 2, 1)
    assert [item["clip_id"] for item in selected] == ["p0", "p1", "w0"]


def test_wilor_manifest_keeps_exact_video_frame_binding(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "signal4d-sign-domain-smplerx-selection-v1",
                "clips": [
                    {
                        "clip_id": "phoenix_clip",
                        "video": str(video),
                        "frame_indices": [3, 7],
                        "source_contract": {"width": 260, "height": 210},
                    }
                ],
            }
        )
    )
    output = tmp_path / "frames.json"
    report = build(selection, output)
    payload = json.loads(output.read_text())
    assert report["frames"] == 2
    assert [row["frame_number"] for row in payload["records"]] == [3, 7]
    assert len(payload["video_sha256"]) == 1


def test_wilor_manifest_shards_are_clip_aligned(tmp_path) -> None:
    source = tmp_path / "frames.json"
    records = []
    for clip in ("clip_a", "clip_b", "clip_c"):
        for frame in range(4):
            records.append(
                {
                    "image_key": f"{clip}_{frame:06d}.png",
                    "video_path": f"/{clip}.mp4",
                    "frame_number": frame,
                }
            )
    source.write_text(
        json.dumps(
            {
                "frame_count": len(records),
                "records": records,
                "video_sha256": {
                    "/clip_a.mp4": "a",
                    "/clip_b.mp4": "b",
                    "/clip_c.mp4": "c",
                },
            }
        )
    )
    report = shard(source, tmp_path / "shards", max_frames=8)
    assert [item["frames"] for item in report["shards"]] == [8, 4]
    first = json.loads((tmp_path / "shards" / "shard_0000.json").read_text())
    assert {_key.rpartition("_")[0] for _key in [r["image_key"][:-4] for r in first["records"]]} == {"clip_a", "clip_b"}


def test_load_hamer_outputs_merges_disjoint_shards(tmp_path) -> None:
    import pickle

    for index, key in enumerate(("a.png", "b.png")):
        path = tmp_path / f"shard_{index:04d}" / "hamer"
        path.mkdir(parents=True)
        with (path / "hamer.pkl").open("wb") as handle:
            pickle.dump({key: index}, handle)
    merged, paths = _load_hamer_outputs(tmp_path)
    assert merged == {"a.png": 0, "b.png": 1}
    assert len(paths) == 2


def test_dexavatar_camera_matches_smplerx_crop_formula() -> None:
    focal, principal = _camera_from_bbox(np.asarray([10, 20, 192, 256]))
    np.testing.assert_allclose(focal, [5000, 5000])
    np.testing.assert_allclose(principal, [106, 148])


def test_dexavatar_hamer_fallback_preserves_real_side() -> None:
    identity = np.eye(3, dtype=np.float32)
    entry = [
        {
            "pred_keypoints_2d": np.zeros((1, 21, 2), dtype=np.float32),
            "pred_keypoints_3d": np.zeros((1, 21, 3), dtype=np.float32),
            "pred_mano_params": {
                "global_orient": identity.reshape(1, 1, 3, 3),
                "hand_pose": np.repeat(identity[None, None], 15, axis=1),
                "betas": np.zeros((1, 10), dtype=np.float32),
            },
        },
        np.zeros((1, 2), dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    ]
    normalized, real = _normalize_hamer_entry(
        entry,
        np.zeros((15, 3), dtype=np.float32),
        np.zeros((15, 3), dtype=np.float32),
        np.zeros((21, 2), dtype=np.float32),
        np.zeros((21, 2), dtype=np.float32),
    )
    assert real == {"left": False, "right": True}
    np.testing.assert_array_equal(normalized[3].astype(int), [1, 0])
    assert normalized[0]["pred_mano_params"]["hand_pose"].shape == (2, 15, 3, 3)


def test_expert_disagreement_is_normalized_geodesic_angle() -> None:
    first = np.zeros((1, 51, 3), dtype=np.float32)
    second = first.copy()
    second[:, 21, 0] = np.pi / 2
    disagreement = _expert_disagreement(first, second)
    assert disagreement.shape == (1, 51)
    np.testing.assert_allclose(disagreement[0, 21], 0.5, atol=1e-5)
    assert np.count_nonzero(disagreement) == 1


def test_wilor_left_hand_uses_dexavatar_canonical_reflection() -> None:
    angle = torch.tensor([[[0.2, 0.3, 0.4]] * 15], dtype=torch.float32)
    from phase2_refiner.geometry.rotations import axis_angle_to_matrix

    rotations = axis_angle_to_matrix(angle)
    entry = [
        {"pred_mano_params": {"hand_pose": rotations}},
        torch.zeros(1, 2),
        torch.ones(1),
        torch.zeros(1),
    ]
    left = _wilor_hands(entry)["left"]
    np.testing.assert_allclose(left[:, 0], 0.2, atol=1e-5)
    np.testing.assert_allclose(left[:, 1], -0.3, atol=1e-5)
    np.testing.assert_allclose(left[:, 2], -0.4, atol=1e-5)
