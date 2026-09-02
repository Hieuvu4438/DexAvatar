import numpy as np

from signdart.geometry.rotations import rotation_between


def test_rotation_between_parallel_and_antiparallel_is_proper():
    for target in (np.asarray([1.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0])):
        rotation = rotation_between(np.asarray([1.0, 0.0, 0.0]), target)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
        assert np.allclose(rotation @ np.asarray([1.0, 0.0, 0.0]), target, atol=1e-8)

