import numpy as np
import pytest
import pickle

from phase2_refiner.data.extract_how2sign_teacher import _load_track
from phase2_refiner.data.prepare_sign_domain_selection import (
    _select_window,
    _wlasl_split,
)


def test_select_window_binds_original_frame_ids_and_annotation_rows() -> None:
    original = np.asarray([0, 1, 3, 4, 8, 9, 10, 14], dtype=np.int64)
    frames, rows = _select_window(original, 4, 123, "clip")
    np.testing.assert_array_equal(frames, original[rows])
    assert len(frames) == 4
    assert np.all(np.diff(rows) == 1)


def test_select_window_rejects_non_monotonic_release_indices() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _select_window(np.asarray([0, 2, 2, 3]), 3, 1, "bad")


def test_wlasl_split_never_uses_official_validation_for_fitting() -> None:
    assert _wlasl_split("11", "val", 12345) is None
    assert _wlasl_split("11", "test", 12345) == "test"
    assert _wlasl_split("11", "train", 12345) in {
        "train",
        "val",
        "calibration",
    }


def test_load_track_selects_strongest_person_from_ragged_frames(tmp_path) -> None:
    first_points = np.zeros((1, 133, 2), dtype=np.float32)
    second_points = np.stack(
        (
            np.full((133, 2), 1.0, dtype=np.float32),
            np.full((133, 2), 2.0, dtype=np.float32),
        )
    )
    path = tmp_path / "ragged.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "keypoints": [first_points, second_points],
                "scores": [
                    np.zeros((1, 133), dtype=np.float32),
                    np.stack(
                        (
                            np.zeros(133, dtype=np.float32),
                            np.ones(133, dtype=np.float32),
                        )
                    ),
                ],
            },
            handle,
        )
    keypoints, scores = _load_track(path)
    assert keypoints.shape == (2, 133, 2)
    assert scores.shape == (2, 133)
    np.testing.assert_array_equal(keypoints[1], second_points[1])
