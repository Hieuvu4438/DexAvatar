from pathlib import Path

import torch

from signpk.data.frame_manifest import FrameRecord, SignManifest
from signpk.observers.omnihands_wrapper import export_omnihands_output, load_omnihands_cache


def test_omnihands_raw_output_roundtrip(tmp_path: Path):
    frames = 2
    rgb = tmp_path / "frame.png"
    rgb.write_bytes(b"x")
    records = tuple(
        FrameRecord(i, i + 1, (i + 1) * 2, (i + 1) * 2, float(i), rgb, None)
        for i in range(frames)
    )
    manifest = SignManifest("S", 1, 2, "~0", "unknown", "x2", "reflect", records)
    joints = torch.zeros(frames, 21, 3)
    joints[:, 5] = torch.tensor([1.0, 1.0, 0.0])
    joints[:, 9] = torch.tensor([0.0, 1.0, 0.0])
    joints[:, 17] = torch.tensor([-1.0, 1.0, 0.0])
    output = {"root_rel": torch.zeros(frames, 3)}
    for side in ("left", "right"):
        output.update(
            {
                f"mano_pose_{side}": torch.zeros(frames, 48),
                f"mano_pose6d_{side}": torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).repeat(frames, 16),
                f"mano_shape_{side}": torch.zeros(frames, 10),
                f"verts3d_{side}": torch.zeros(frames, 778, 3),
                f"joints3d_{side}": joints,
                f"verts3d_world_{side}": torch.zeros(frames, 778, 3),
                f"joints3d_world_{side}": joints,
                f"cam_aligned_{side}": torch.zeros(frames, 3),
            }
        )
    windows = [
        {"center_index": i, "indices": [0, 1, 0], "padded": [True, False, True]}
        for i in range(frames)
    ]
    path = tmp_path / "omni.pt"
    export_omnihands_output(
        output,
        torch.zeros(frames, 2, 1024),
        manifest,
        windows,
        {"left": torch.ones(frames, 4), "right": torch.ones(frames, 4)},
        {"left": torch.ones(frames, dtype=torch.bool), "right": torch.ones(frames, dtype=torch.bool)},
        path,
        {"commit": "test"},
    )
    left, right, root_rel, metadata = load_omnihands_cache(path, manifest)
    assert left.pose_rotmat.shape == (frames, 16, 3, 3)
    assert right.temporal_token.shape == (frames, 1024)
    torch.testing.assert_close(left.padding_ratio, torch.full((frames,), 2 / 3))
    assert root_rel.shape == (frames, 3)
    assert metadata["commit"] == "test"
