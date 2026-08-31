import json
from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.cache_schema import CacheClip
from signal4d_external.prepare_synth3d_targets import (
    Annotation,
    attach_synth3d_target,
    target_frame_indices,
)


def _clip() -> CacheClip:
    frames = 2
    return CacheClip(
        clip_id="clip",
        frame_names=np.asarray(["frame0", "frame2"]),
        init_axis_angle=np.zeros((frames, 51, 3), np.float32),
        observation_features=np.zeros((frames, 51, 8), np.float32),
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
        source_paths=np.asarray(["/external/frame0", "/external/frame2"]),
        target_axis_angle=np.ones((frames, 51, 3), np.float32),
        target_rotation_valid=np.ones((frames, 51), bool),
        frame_numbers=np.asarray([0, 2], np.int64),
        fps=10.0,
        metadata_json=json.dumps(
            {
                "dataset": "How2Sign",
                "source_group": "sequence",
                "target_provider": "old pseudo target",
                "target_type": "independent_pseudo_target",
                "sgnify_training_reads": 0,
            }
        ),
    )


def _annotation() -> Annotation:
    return Annotation(
        sentence_name="sequence_0-5-rgb_front",
        video_name="sequence-5-rgb_front",
        start=1.05,
        end=2.0,
        aligned=True,
    )


def _fit(frames: int = 20) -> dict[str, np.ndarray]:
    theta = np.zeros((frames, 52, 3), dtype=np.float32)
    theta[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None]
    return {"thetas": theta}


def test_target_binding_uses_aligned_start_fps_and_local_frame_number() -> None:
    assert target_frame_indices(_clip(), _annotation()).tolist() == [11, 13]


def test_attach_replaces_only_target_and_records_external_lineage(tmp_path: Path) -> None:
    clip = _clip()
    updated = attach_synth3d_target(
        clip,
        _annotation(),
        _fit(),
        fit_path=tmp_path / "sequence-5.npz",
        fit_sha256="a" * 64,
        annotation_path=tmp_path / "how2sign_train_aligned.csv",
        annotation_sha256="b" * 64,
        offsets_sha256="c" * 64,
        phase2_split="train",
    )
    np.testing.assert_array_equal(updated.init_axis_angle, clip.init_axis_angle)
    np.testing.assert_array_equal(updated.observation_features, clip.observation_features)
    assert updated.target_axis_angle[:, 0, 0].tolist() == [11.0, 13.0]
    assert updated.target_rotation_valid.all()
    metadata = json.loads(updated.metadata_json)
    assert metadata["target_dataset"] == "How2Sign-Synth3D"
    assert metadata["sgnify_training_reads"] == 0
    assert metadata["target_contract"]["first_fit_frame"] == 11
    assert metadata["previous_target_provider"] == "old pseudo target"


def test_attach_fails_closed_when_fit_is_too_short(tmp_path: Path) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        attach_synth3d_target(
            _clip(),
            _annotation(),
            _fit(frames=12),
            fit_path=tmp_path / "sequence-5.npz",
            fit_sha256="a" * 64,
            annotation_path=tmp_path / "how2sign_train_aligned.csv",
            annotation_sha256="b" * 64,
            offsets_sha256="c" * 64,
            phase2_split="train",
        )
