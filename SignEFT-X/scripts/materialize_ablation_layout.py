#!/usr/bin/env python3
"""Create a zero-copy official-evaluator layout for an ablation mesh tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from signeft.io.obj import load_obj, validate_mesh
from signeft.io_utils import atomic_write_json, sha256_file
from signeft.manifest import read_hand_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_hand_manifest(args.manifest)
    counts: dict[str, int] = {}
    for record in records:
        source = (
            args.mesh_root / record.sign / f"{record.source_frame_id:06d}.obj"
        ).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = (
            args.output / record.sign / "smplifyx" / "meshes"
            / f"{record.frame_index:03d}.obj"
        )
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"different layout entry exists: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
        counts[record.sign] = counts.get(record.sign, 0) + 1
    if records:
        vertices, faces = load_obj(
            args.output / records[0].sign / "smplifyx" / "meshes" / "000.obj"
        )
        validate_mesh(vertices, faces)
    atomic_write_json(args.output / "materialization_summary.json", {
        "schema_version": "signeft.ablation-evaluation-layout.v1",
        "layout": "absolute_symlinks",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "mesh_root": str(args.mesh_root.resolve()),
        "signs": len(counts),
        "frames": len(records),
        "items": [
            {"sign": sign, "frames": count}
            for sign, count in sorted(counts.items())
        ],
    })


if __name__ == "__main__":
    main()
