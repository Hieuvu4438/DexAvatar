"""Freeze the exact licensed selfcontact files approved for a DCG run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dcg_sign4d.utils.hashing import file_sha256


def build_registry(
    essentials_root: str | Path,
    source_root: str | Path,
    expected_commit: str,
    output: str | Path,
) -> Path:
    root = Path(essentials_root).resolve()
    source = Path(source_root).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable selfcontact registry exists: {output}")
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise ValueError("selfcontact source commit mismatch")
    required = [
        Path("geodesics/smplx/smplx_neutral_geodesic_dist.npy"),
        Path("hd_model/smplx/smplx_neutral_hd_sample_from_mesh_out.pkl"),
        Path("hd_model/smplx/smplx_neutral_hd_vert_regressor_sparse.npz"),
        Path("models_utils/smplx/smplx_faces.npy"),
        *[path.relative_to(root) for path in sorted((root / "segments/smplx").glob("*"))],
    ]
    files = []
    for relative in required:
        path = root / relative
        if not path.is_file() or path.name.startswith("._"):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    required_names = {
        "smplx_neutral_geodesic_dist.npy",
        "smplx_neutral_hd_sample_from_mesh_out.pkl",
        "smplx_neutral_hd_vert_regressor_sparse.npz",
        "smplx_faces.npy",
        "smplx_segments_bounds.pkl",
        "smplx_inner_mouth_bounds.pkl",
    }
    observed_names = {Path(row["path"]).name for row in files}
    if not required_names <= observed_names or not any(
        name.startswith("smplx_segment_") and name.endswith(".ply") for name in observed_names
    ):
        raise FileNotFoundError("selfcontact essentials snapshot is incomplete")
    payload = {
        "schema_version": "dcg_selfcontact_essentials_v1",
        "scientific_status": "FROZEN",
        "authorization_basis": "user-approved local licensed asset snapshot",
        "source_commit": commit,
        "essentials_root": str(root),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essentials-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = build_registry(
        args.essentials_root,
        args.source_root,
        args.expected_commit,
        args.output,
    )
    print(json.dumps({"path": str(path.resolve()), "sha256": file_sha256(path)}))


if __name__ == "__main__":
    main()
