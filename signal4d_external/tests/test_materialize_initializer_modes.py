from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from phase2_refiner.data.add_reprojection_residuals import _camera_matrix
from phase2_refiner.data.cache_schema import (
    PHASE2R_SEMANTIC_CONTRACT,
    CacheClip,
    validate_phase2r_semantics,
)
from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase2_refiner.provenance import sha256_file
from signal4d_external.materialize_initializer import _replace_raw_initializer


def _template() -> CacheClip:
    frames = 2
    observations = np.zeros((frames, 51, 8), np.float32)
    observations[..., 1] = 1.0
    clip = CacheClip(
        clip_id="Ablehnen",
        frame_names=np.asarray(["low_149", "low_151"]),
        frame_numbers=np.asarray([149, 151]),
        fps=25.0,
        init_axis_angle=np.zeros((frames, 51, 3), np.float32),
        observation_features=observations,
        keypoints_2d=np.zeros((frames, 51, 2), np.float32),
        keypoint_valid=np.ones((frames, 51), bool),
        refine_mask=np.ones(51, bool),
        betas=np.zeros(10, np.float32),
        global_orient=np.zeros((frames, 3), np.float32),
        transl=np.zeros((frames, 3), np.float32),
        jaw_pose=np.zeros((frames, 3), np.float32),
        leye_pose=np.zeros((frames, 3), np.float32),
        reye_pose=np.zeros((frames, 3), np.float32),
        expression=np.zeros((frames, 10), np.float32),
        source_paths=np.asarray(["unused-149.pkl", "unused-151.pkl"]),
        metadata_json=json.dumps(
            {"coordinate_policy": {"keypoints_2d": "normalized_image_-1_to_1"}}
        ),
    )
    clip.validate()
    return clip


def _params(value: float) -> dict[str, np.ndarray]:
    return {
        "betas": np.full(10, value, np.float32),
        "global_orient": np.full(3, value, np.float32),
        "transl": np.full(3, value, np.float32),
        "jaw_pose": np.full(3, value, np.float32),
        "leye_pose": np.full(3, value, np.float32),
        "reye_pose": np.full(3, value, np.float32),
        "expression": np.full(10, value, np.float32),
        "focal": np.asarray([1000.0, 1001.0], np.float32),
        "princpt": np.asarray([250.0, 150.0], np.float32),
    }


def test_raw_mode_uses_smplerx_body_wilor_hands_and_side_fallback(
    tmp_path: Path,
) -> None:
    template = _template()
    smplerx_root = tmp_path / "smplerx"
    params_dir = smplerx_root / "Ablehnen" / "smplerx" / "smplx"
    params_dir.mkdir(parents=True)
    params_paths = []
    for frame, value in ((149, 0.1), (151, 0.3)):
        path = params_dir / f"low_{frame:03d}.pkl"
        with path.open("wb") as handle:
            pickle.dump(_params(value), handle, protocol=2)
        params_paths.append(path)

    rotations = torch.eye(3).expand(2, 2, 55, 3, 3).clone()
    valid = torch.zeros((2, 2, 55), dtype=torch.bool)
    body_axis = torch.zeros((2, 21, 3))
    body_axis[..., 2] = 0.1
    smplerx_left = torch.zeros((2, 15, 3))
    smplerx_left[..., 0] = 0.2
    smplerx_right = torch.zeros((2, 15, 3))
    smplerx_right[..., 0] = -0.2
    wilor_left = torch.zeros((2, 15, 3))
    wilor_left[..., 1] = 0.3
    wilor_right = torch.zeros((2, 15, 3))
    wilor_right[..., 2] = 0.4
    rotations[:, 0, 1:22] = axis_angle_to_matrix(body_axis)
    rotations[:, 0, 25:40] = axis_angle_to_matrix(smplerx_left)
    rotations[:, 0, 40:55] = axis_angle_to_matrix(smplerx_right)
    rotations[:, 1, 25:40] = axis_angle_to_matrix(wilor_left)
    rotations[:, 1, 40:55] = axis_angle_to_matrix(wilor_right)
    valid[:, 0] = True
    valid[0, 1, 25:40] = True
    valid[:, 1, 40:55] = True

    observation_dir = tmp_path / "observations" / "Ablehnen"
    observation_dir.mkdir(parents=True)
    tensor_path = observation_dir / "observations.safetensors"
    save_file(
        {
            "frame_ids": torch.tensor([149, 151], dtype=torch.int64),
            "rotations": rotations,
            "valid_rot": valid,
        },
        tensor_path,
    )
    metadata_path = observation_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_sha256": sha256_file(tensor_path),
                "sources": [
                    {"source_id": 0, "name": "smplerx", "role": "body_initializer"},
                    {"source_id": 1, "name": "wilor", "role": "hand_hypothesis"},
                ],
                "source_hashes": {
                    str(path.resolve()): sha256_file(path) for path in params_paths
                },
            }
        ),
        encoding="utf-8",
    )

    result = _replace_raw_initializer(
        template,
        observation_dir,
        smplerx_root,
        Path("smplerx/smplx"),
    )
    result_matrix = axis_angle_to_matrix(torch.from_numpy(result.init_axis_angle))
    torch.testing.assert_close(result_matrix[:, :21], rotations[:, 0, 1:22])
    torch.testing.assert_close(result_matrix[0, 21:36], rotations[0, 1, 25:40])
    torch.testing.assert_close(result_matrix[1, 21:36], rotations[1, 0, 25:40])
    torch.testing.assert_close(result_matrix[:, 36:51], rotations[:, 1, 40:55])
    assert result.fallback_reason.tolist() == ["", "lhand_smplerx_fallback"]
    assert result.initializer_component.tolist() == [
        "body=smplerx;hands=wilor",
        "body=smplerx;hands=wilor",
    ]
    assert not result.alternate_rotation_valid.any()
    np.testing.assert_allclose(result.betas, 0.2)
    assert json.loads(result.metadata_json)["sgnify_target_reads"] == 0
    assert result.semantic_contract_version == PHASE2R_SEMANTIC_CONTRACT
    validate_phase2r_semantics(result)


def test_reprojection_camera_accepts_fitted_and_raw_smplerx_formats() -> None:
    fitted = np.asarray([[2.0, 0.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(_camera_matrix({"K": fitted}, "fitted"), fitted)
    raw = _camera_matrix(
        {
            "focal": np.asarray([1000.0, 1001.0]),
            "princpt": np.asarray([250.0, 150.0]),
        },
        "raw",
    )
    np.testing.assert_array_equal(
        raw,
        np.asarray(
            [[1000.0, 0.0, 250.0], [0.0, 1001.0, 150.0], [0.0, 0.0, 1.0]],
            np.float32,
        ),
    )
