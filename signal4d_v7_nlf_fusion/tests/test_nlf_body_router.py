import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).parents[1] / "nlf_body_router.py"
SPEC = importlib.util.spec_from_file_location("nlf_body_router", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_geodesic_blend_endpoints():
    first = np.eye(3, dtype=np.float32)[None]
    second = Rotation.from_rotvec([[0.0, 0.0, 0.7]]).as_matrix().astype(np.float32)
    assert np.allclose(MODULE.geodesic_blend(first, second, 0.0), first, atol=1e-6)
    assert np.allclose(MODULE.geodesic_blend(first, second, 1.0), second, atol=1e-6)


def test_preserve_global_wrist_rotation_after_parent_change():
    parents = np.asarray([-1, 0, 1], dtype=np.int64)
    reference = Rotation.from_rotvec(
        [[0.0, 0.0, 0.1], [0.0, 0.2, 0.0], [0.3, 0.0, 0.0]]
    ).as_matrix().astype(np.float32)
    candidate = reference.copy()
    candidate[1] = Rotation.from_rotvec([0.0, -0.6, 0.0]).as_matrix()
    preserved = MODULE.preserve_global_rotations(reference, candidate, parents, (2,))
    reference_global = MODULE.global_rotations(reference, parents)
    preserved_global = MODULE.global_rotations(preserved, parents)
    assert np.allclose(preserved_global[2], reference_global[2], atol=1e-6)
    assert np.allclose(preserved[1], candidate[1], atol=1e-6)
