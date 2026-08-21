import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "nlf_gtfree_2d_temporal_gate.py"
SPEC = importlib.util.spec_from_file_location("nlf_gtfree_2d_temporal_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_projection_uses_camera_intrinsics_and_image_diagonal():
    joints = np.asarray([[1.0, -2.0, -10.0]], dtype=np.float32)
    camera = np.asarray([[100.0, 0.0, 50.0], [0.0, 200.0, 40.0], [0.0, 0.0, 1.0]])
    result = MODULE.project_normalized(joints, camera, np.asarray([60, 80]))
    assert np.allclose(result, [[0.6, 0.8]])


def test_viterbi_prefers_candidate_when_position_and_motion_are_better():
    observed = np.asarray([[[0.0, 0.0]], [[0.1, 0.0]], [[0.2, 0.0]]])
    projected = np.zeros((3, 2, 1, 2), dtype=np.float64)
    projected[:, 0, 0, 0] = [0.3, 0.3, 0.3]
    projected[:, 1, 0, 0] = [0.0, 0.1, 0.2]
    selected, _, _ = MODULE.viterbi_select(
        observed, projected, np.ones((3, 1)), np.ones(3, bool), 1, 1.0
    )
    assert selected.tolist() == [True, True, True]


def test_invalid_candidate_and_exact_tie_fall_back_to_v6():
    observed = np.zeros((2, 2, 2), dtype=np.float64)
    projected = np.zeros((2, 2, 2, 2), dtype=np.float64)
    selected, _, _ = MODULE.viterbi_select(
        observed, projected, np.ones((2, 2)), np.asarray([True, False]), 2, 1.0
    )
    assert selected.tolist() == [False, False]
