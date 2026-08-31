import numpy as np
import pytest

from signpccx.schema import REQUIRED, validate_npz


def valid_arrays():
    return {key: np.zeros(shape, dtype=np.float32) for key, shape in REQUIRED.items()}


def test_teacher_schema_accepts_exact_contract(tmp_path):
    path = tmp_path / "teacher.npz"
    np.savez(path, **valid_arrays())
    validate_npz(path)


def test_teacher_schema_rejects_missing_field(tmp_path):
    arrays = valid_arrays()
    arrays.pop("K_full")
    path = tmp_path / "teacher.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="missing"):
        validate_npz(path)


def test_teacher_schema_rejects_nan(tmp_path):
    arrays = valid_arrays()
    arrays["smplx_vertices_cam"][0, 0] = np.nan
    path = tmp_path / "teacher.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_npz(path)

