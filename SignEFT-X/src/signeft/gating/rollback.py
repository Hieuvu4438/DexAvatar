from __future__ import annotations

from pathlib import Path
import shutil

from signeft.io_utils import sha256_file


def exact_rollback(
    baseline_obj: Path,
    baseline_state: Path,
    output_obj: Path,
    output_state: Path,
) -> dict[str, str]:
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    output_state.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline_obj, output_obj)
    shutil.copyfile(baseline_state, output_state)
    hashes = {
        "baseline_obj": sha256_file(baseline_obj),
        "output_obj": sha256_file(output_obj),
        "baseline_state": sha256_file(baseline_state),
        "output_state": sha256_file(output_state),
    }
    if hashes["baseline_obj"] != hashes["output_obj"]:
        raise RuntimeError("exact OBJ rollback hash mismatch")
    if hashes["baseline_state"] != hashes["output_state"]:
        raise RuntimeError("exact state rollback hash mismatch")
    return hashes

