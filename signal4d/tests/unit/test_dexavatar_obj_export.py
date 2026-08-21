from __future__ import annotations

from pathlib import Path

import numpy as np

from signal4d.io.obj import read_simple_obj, write_dexavatar_obj


def test_dexavatar_obj_roundtrip_and_dialect(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    vertices = rng.normal(size=(10475, 3)).astype(np.float32)
    faces = np.column_stack(
        [
            np.arange(20908) % 10475,
            (np.arange(20908) + 1) % 10475,
            (np.arange(20908) + 2) % 10475,
        ]
    )
    target = tmp_path / "low_149.obj"
    write_dexavatar_obj(target, vertices, faces)
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# https://github.com/mikedh/trimesh\nv ")
    assert text.endswith("\n\n")
    restored_vertices, restored_faces = read_simple_obj(target)
    np.testing.assert_array_equal(restored_faces, faces)
    assert np.max(np.abs(restored_vertices - vertices)) <= 5.1e-9
