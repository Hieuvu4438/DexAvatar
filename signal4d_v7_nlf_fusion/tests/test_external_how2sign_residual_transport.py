import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).parents[1] / "external_how2sign_residual_transport.py"
SPEC = importlib.util.spec_from_file_location("external_how2sign_residual_transport", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_transport_is_identity_at_zero_alpha():
    reference = Rotation.from_rotvec([[0.2, -0.1, 0.3]]).as_matrix()
    initial = Rotation.from_rotvec([[-0.2, 0.4, 0.1]]).as_matrix()
    refined = Rotation.from_rotvec([[0.1, 0.5, -0.2]]).as_matrix()
    output = MODULE.transport_local_residual(reference, initial, refined, 0.0)
    assert np.allclose(output, reference, atol=1e-6)


def test_full_transport_preserves_source_relative_rotation():
    reference = Rotation.from_rotvec([[0.2, -0.1, 0.3]]).as_matrix()
    initial = Rotation.from_rotvec([[-0.2, 0.4, 0.1]]).as_matrix()
    refined = Rotation.from_rotvec([[0.1, 0.5, -0.2]]).as_matrix()
    output = MODULE.transport_local_residual(reference, initial, refined, 1.0)
    source_delta = refined @ np.swapaxes(initial, -1, -2)
    output_delta = output @ np.swapaxes(reference, -1, -2)
    assert np.allclose(output_delta, source_delta, atol=1e-6)


def test_pose51_matches_smplx_body_and_hand_contract():
    params = {
        "body_pose": np.arange(63),
        "left_hand_pose": np.arange(45) + 100,
        "right_hand_pose": np.arange(45) + 200,
    }
    pose = MODULE.pose51(params)
    assert pose.shape == (51, 3)
    assert pose[0, 0] == 0
    assert pose[21, 0] == 100
    assert pose[36, 0] == 200
