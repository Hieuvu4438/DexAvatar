from __future__ import annotations

from pathlib import Path

import numpy as np

from signal4d.evaluation.author_sgnify import (
    _expected_central_frame_ids,
    _load_author_module,
    _read_sign_classes,
)

REPOSITORY = Path(__file__).resolve().parents[3]
AUTHOR_ROOT = REPOSITORY / "data" / "evaluation_from_author"


def test_author_metric_functions_are_loaded_from_supplied_source() -> None:
    author = _load_author_module(AUTHOR_ROOT / "evaluate_new_fitting.py")
    source = np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    target = np.asarray([[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
    np.testing.assert_allclose(author.transl_point_error(source, target), 0.0)


def test_author_sign_classes_and_central_frames_are_bound_to_author_assets() -> None:
    classes = _read_sign_classes(AUTHOR_ROOT / "signs.txt")
    segments = {"Ablehnen": [149, 175]}
    frames = _expected_central_frame_ids(
        "Ablehnen", REPOSITORY / "data" / "smplx_gt", segments
    )
    assert classes["Ablehnen"] == "~0"
    assert frames == list(range(149, 176, 2))
