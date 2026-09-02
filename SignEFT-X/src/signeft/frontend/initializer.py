"""Construct a deterministic full-coverage view of frozen initializers."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

from signeft.io_utils import atomic_write_json, sha256_file


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def build_initializer_view(
    manifest: Path,
    primary: Path,
    fallback: Path,
    output: Path,
) -> dict[str, object]:
    """Use a primary reconstruction when complete, else one whole fallback frame."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output}")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("initializer manifest is empty")
    identities = [(row["sign"], Path(row["prediction_path"]).stem) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("initializer manifest has duplicate sign/frame identities")
    counts: Counter[str] = Counter()
    selections = []
    for sign, frame in identities:
        result = Path(sign) / "smplifyx/results" / f"{frame}.pkl"
        mesh = Path(sign) / "smplifyx/meshes" / f"{frame}.obj"
        use_primary = (primary / result).is_file() and (primary / mesh).is_file()
        source = primary if use_primary else fallback
        label = "primary" if use_primary else "fallback"
        _link(source / result, output / result)
        _link(source / mesh, output / mesh)
        counts[label] += 1
        selections.append({"sign": sign, "frame": frame, "source": label})
    report = {
        "schema_version": "signeft.initializer-view.v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "primary": str(primary.resolve()),
        "fallback": str(fallback.resolve()),
        "output": str(output.resolve()),
        "frames": len(rows),
        "primary_frames": counts["primary"],
        "fallback_frames": counts["fallback"],
        "fallback_fraction": counts["fallback"] / len(rows),
        "selection": selections,
    }
    atomic_write_json(output / "locked_view_manifest.json", report)
    return report
