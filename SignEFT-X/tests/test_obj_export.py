from pathlib import Path

import numpy as np

from signeft.io.obj import load_obj, write_obj


def test_obj_writer_preserves_order_and_nine_decimal_precision(tmp_path: Path):
    vertices = np.zeros((10475, 3), dtype=np.float64)
    vertices[0] = (0.1234567894, -0.9876543214, 1.0 / 3.0)
    vertices[-1] = (-1.25, 2.5, -3.75)
    faces = np.zeros((20908, 3), dtype=np.int64)
    faces[0] = (0, 1, 2)
    faces[-1] = (10472, 10473, 10474)
    path = tmp_path / "mesh.obj"
    write_obj(path, vertices, faces)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "v 0.123456789 -0.987654321 0.333333333"
    assert lines[10474] == "v -1.250000000 2.500000000 -3.750000000"
    assert lines[10475] == "f 1 2 3"
    assert lines[-1] == "f 10473 10474 10475"
    restored_vertices, restored_faces = load_obj(path)
    assert np.max(np.abs(restored_vertices - vertices)) <= 5.1e-10
    assert np.array_equal(restored_faces, faces)
