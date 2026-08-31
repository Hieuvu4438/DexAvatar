import numpy as np
import pytest

from signpccx.export.obj import EXPORT_TRANSFORMS, validate_mesh, write_obj_atomic
from signpccx.export.preflight import load_obj_minimal, preflight_sign


def tiny_mesh():
    vertices = np.zeros((10475, 3), dtype=np.float32)
    vertices[0] = [1.0, 2.0, 3.0]
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    return vertices, faces


def test_export_x180_and_identity(tmp_path):
    vertices, faces = tiny_mesh()
    x180 = tmp_path / "x180.obj"
    identity = tmp_path / "identity.obj"
    write_obj_atomic(x180, vertices, faces, "x180")
    write_obj_atomic(identity, vertices, faces, "identity")
    x_vertices, _ = load_obj_minimal(x180)
    i_vertices, _ = load_obj_minimal(identity)
    np.testing.assert_array_equal(x_vertices[0], [1.0, -2.0, -3.0])
    np.testing.assert_array_equal(i_vertices[0], [1.0, 2.0, 3.0])


def test_validate_mesh_rejects_nan():
    vertices, faces = tiny_mesh()
    vertices[4, 0] = np.nan
    with pytest.raises(FloatingPointError):
        validate_mesh(vertices, faces)


def test_preflight_rejects_non_contiguous_names(tmp_path):
    vertices, faces = tiny_mesh()
    write_obj_atomic(tmp_path / "001.obj", vertices, faces, "identity")
    with pytest.raises(RuntimeError, match="contiguous"):
        preflight_sign(tmp_path, 1, faces)

