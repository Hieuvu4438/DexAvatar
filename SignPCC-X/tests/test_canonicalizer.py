from pathlib import Path
import pickle

import numpy as np

from signpccx.model.canonicalizer import (
    load_external_parameters,
    load_mano_smplx_ids,
    load_obj_vertices,
)


def test_external_parameter_schema(tmp_path: Path):
    shapes = {
        "betas": 10, "global_orient": 3, "body_pose": 63,
        "left_hand_pose": 45, "right_hand_pose": 45, "jaw_pose": 3,
        "leye_pose": 3, "reye_pose": 3, "expression": 10, "transl": 3,
    }
    path = tmp_path / "result.pkl"
    with path.open("wb") as handle:
        pickle.dump({key: np.zeros((1, size), np.float32) for key, size in shapes.items()}, handle)
    result = load_external_parameters(path)
    assert result["body_pose"].shape == (63,)
    assert all(np.isfinite(value).all() for value in result.values())


def test_obj_vertex_loader(tmp_path: Path):
    path = tmp_path / "mesh.obj"
    path.write_text("v 1 2 3\nv 4 5 6\nf 1 2 1\n", encoding="utf-8")
    assert np.array_equal(load_obj_vertices(path, expected=2), [[1, 2, 3], [4, 5, 6]])


def test_mano_mapping_validation(tmp_path: Path):
    path = tmp_path / "ids.pkl"
    with path.open("wb") as handle:
        pickle.dump({"left_hand": np.arange(778), "right_hand": np.arange(778, 1556)}, handle)
    left, right = load_mano_smplx_ids(path)
    assert len(left) == len(right) == 778
    assert not np.intersect1d(left, right).size
