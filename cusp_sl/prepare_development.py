"""Freeze a deterministic How2Sign validation subset for energy calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phase2_refiner.data.cache_schema import load_cache_clip


def stable_key(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.name}".encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    entries = payload.get("clips", payload)
    paths = [
        (args.source_manifest.parent / entry).resolve()
        if not Path(entry).is_absolute() else Path(entry)
        for entry in entries
    ]
    paths.sort(key=lambda path: stable_key(path, args.seed))
    if args.clips < 1 or args.clips > len(paths):
        raise ValueError(f"Requested {args.clips} clips from {len(paths)}")
    selected = paths[:args.clips]
    groups, frames = set(), 0
    for path in selected:
        clip = load_cache_clip(path)
        metadata = json.loads(clip.metadata_json)
        group = metadata.get("source_group")
        if not group:
            raise ValueError(f"Missing source_group in {path}")
        groups.add(str(group))
        frames += len(clip.frame_names)
    report = {
        "role": "development_validation_no_sgnify",
        "seed": args.seed,
        "clips": [str(path) for path in selected],
        "clip_count": len(selected),
        "frame_count": frames,
        "source_groups": len(groups),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "clips"}, indent=2))


if __name__ == "__main__":
    main()
