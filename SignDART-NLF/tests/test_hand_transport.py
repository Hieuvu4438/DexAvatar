import numpy as np

from signdart.model import rigid_transport_hand_vertices


def test_rigid_hand_transport_preserves_centered_geometry():
    incumbent_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    candidate_vertices = incumbent_vertices + 7.0
    incumbent_joints = np.zeros((22, 3), dtype=np.float32)
    candidate_joints = incumbent_joints.copy()
    candidate_joints[20] = [0.2, -0.3, 0.4]
    result = rigid_transport_hand_vertices(
        candidate_vertices,
        candidate_joints,
        incumbent_vertices,
        incumbent_joints,
        np.asarray([0, 1, 2]),
        20,
    )
    centered = result - result.mean(axis=0, keepdims=True)
    incumbent_centered = incumbent_vertices - incumbent_vertices.mean(axis=0, keepdims=True)
    assert np.allclose(centered, incumbent_centered, atol=1e-7)
