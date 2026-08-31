"""Create immutable target-free cache copies for development inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_targets(clip, *, source_path: Path, source_hash: str):
    metadata = json.loads(clip.metadata_json)
    metadata["development_inference_contract"] = {
        "target_arrays_removed": True,
        "target_quality_zeroed": True,
        "source_cache": str(source_path.resolve()),
        "source_cache_sha256": source_hash,
    }
    return replace(
        clip,
        target_axis_angle=None,
        target_rotation_valid=None,
        target_joint_positions=None,
        target_joint_valid=None,
        target_quality=np.zeros_like(clip.target_quality),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    (args.output / "clips").mkdir(parents=True)
    entries = []
    frames = 0
    summaries = []
    for entry in source["clips"]:
        relative = entry["cache"] if isinstance(entry, dict) else entry
        source_path = Path(relative)
        if not source_path.is_absolute():
            source_path = args.manifest.parent / source_path
        source_hash = sha256(source_path)
        clip = strip_targets(
            load_cache_clip(source_path),
            source_path=source_path,
            source_hash=source_hash,
        )
        destination = args.output / "clips" / f"{clip.clip_id}.npz"
        save_cache_clip(destination, clip)
        entries.append(str(Path("clips") / destination.name))
        frames += len(clip.frame_names)
        summaries.append({
            "clip_id": clip.clip_id,
            "frames": len(clip.frame_names),
            "source_cache_sha256": source_hash,
            "targetless_cache_sha256": sha256(destination),
        })
    report = {
        "role": "development_targetless_inference",
        "target_reads_permitted": 0,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "clips": entries,
        "clip_count": len(entries),
        "expected_frames": frames,
        "summaries": summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"clips": len(entries), "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
