from pathlib import Path

from signeft.gating.rollback import exact_rollback


def test_exact_rollback_is_byte_identical(tmp_path: Path):
    base_obj = tmp_path / "base.obj"
    base_state = tmp_path / "base.npz"
    base_obj.write_bytes(b"v 0.000000000 0.000000000 0.000000000\n")
    base_state.write_bytes(bytes(range(255)))
    hashes = exact_rollback(
        base_obj, base_state, tmp_path / "out" / "mesh.obj", tmp_path / "out" / "state.npz"
    )
    assert hashes["baseline_obj"] == hashes["output_obj"]
    assert hashes["baseline_state"] == hashes["output_state"]

