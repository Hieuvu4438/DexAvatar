"""Create a non-destructive, full-coverage initializer view on a locked manifest.

Each frame is taken wholly from the primary method only when both its parameter
PKL and rendered mesh exist.  Otherwise both files come from the fallback.  The
view contains symlinks, so it neither copies nor mutates the frozen methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from phase2_refiner.provenance import sha256_file


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def build_view(
    manifest: Path,
    primary: Path,
    fallback: Path,
    output: Path,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {output}")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Locked manifest is empty")
    ids = [(row["sign"], Path(row["prediction_path"]).stem) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Locked manifest contains duplicate sign/frame IDs")

    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    selections = []
    for sign, frame in ids:
        relative_pkl = Path(sign) / "smplifyx" / "results" / f"{frame}.pkl"
        relative_mesh = Path(sign) / "smplifyx" / "meshes" / f"{frame}.obj"
        primary_complete = (primary / relative_pkl).is_file() and (
            primary / relative_mesh
        ).is_file()
        source_root = primary if primary_complete else fallback
        source_name = "primary" if primary_complete else "fallback"
        _link(source_root / relative_pkl, output / relative_pkl)
        _link(source_root / relative_mesh, output / relative_mesh)
        counts[source_name] += 1
        selections.append(
            {"sign": sign, "frame": frame, "source": source_name}
        )

    # Observation-cache metadata are method-level inputs. Prefer primary and
    # fall back independently; they are never mixed within a frame result.
    for sign in sorted({sign for sign, _ in ids}):
        for relative in (Path("sapiens.pkl"), Path("hamer") / "hamer.pkl"):
            candidate = primary / sign / relative
            if not candidate.is_file():
                candidate = fallback / sign / relative
            _link(candidate, output / sign / relative)

    report = {
        "schema_version": 1,
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
    report_path = output / "locked_view_manifest.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_view(
        args.manifest.resolve(),
        args.primary.resolve(),
        args.fallback.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({key: value for key, value in report.items() if key != "selection"}, indent=2))


if __name__ == "__main__":
    main()
