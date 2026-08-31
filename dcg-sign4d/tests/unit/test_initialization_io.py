import hashlib
import pickle

import numpy as np
import pytest
import torch

from dcg_sign4d.initialization.dexavatar_adapter import DexAvatarPklInitializer
from dcg_sign4d.initialization.trajectory_io import load_trajectory, save_trajectory
from dcg_sign4d.synthetic import make_state


def record(frame):
    return {
        "betas": np.full((1, 10), frame * 0.01, np.float32),
        "global_orient": np.zeros((1, 3), np.float32),
        "body_pose": np.zeros((1, 63), np.float32),
        "transl": np.array([[frame, 0, 0]], np.float32) / 30,
        "left_hand_pose": np.zeros((1, 45), np.float32),
        "right_hand_pose": np.zeros((1, 45), np.float32),
        "jaw_pose": np.zeros((1, 3), np.float32),
        "leye_pose": np.zeros((1, 3), np.float32),
        "reye_pose": np.zeros((1, 3), np.float32),
        "expression": np.zeros((1, 10), np.float32),
        "K": np.eye(3, dtype=np.float32),
    }


def test_trusted_dexavatar_conversion_and_safe_replay(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    hashes = {}
    for frame in (2, 4):
        path = source / f"low_{frame}.pkl"
        path.write_bytes(pickle.dumps(record(frame)))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        DexAvatarPklInitializer(30).reconstruct_from_directory(source, expected_hashes=hashes)
    state, metadata = DexAvatarPklInitializer(30).reconstruct_from_directory(
        source, expected_hashes=hashes, trusted=True
    )
    assert metadata["frame_ids"] == [2, 4]
    assert torch.allclose(state.root_velocity[:, 1, 0], torch.tensor([2.0]))
    save_trajectory(state, tmp_path / "artifact", metadata)
    restored, restored_metadata = load_trajectory(tmp_path / "artifact")
    assert torch.equal(restored.body_rot6d, state.body_rot6d)
    assert restored_metadata["trajectory_sha256"]


def test_trajectory_artifact_is_immutable(tmp_path):
    state = make_state()
    save_trajectory(state, tmp_path / "artifact", {})
    with pytest.raises(FileExistsError):
        save_trajectory(state, tmp_path / "artifact", {})
