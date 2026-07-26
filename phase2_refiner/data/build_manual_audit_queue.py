"""Create a deterministic, source-group-stratified 100-clip review queue."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file


def build_queue(manifest: Path, output: Path, samples: int, seed: int) -> dict:
    records = []
    seen_groups = set()
    candidates = []
    for path in _manifest_paths(manifest):
        clip = load_cache_clip(path)
        metadata = json.loads(clip.metadata_json)
        group = str(metadata.get("source_group", clip.clip_id))
        source = str(clip.source_paths[len(clip.source_paths) // 2]).split("#", 1)[0]
        candidates.append((group, clip, path, source, metadata))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for group, clip, path, source, metadata in candidates:
        if group in seen_groups:
            continue
        seen_groups.add(group)
        records.append(
            {
                "clip_id": clip.clip_id,
                "source_group": group,
                "frames": len(clip.frame_names),
                "video_path": source,
                "cache_path": str(path),
                "target_type": metadata.get("target_type", ""),
                "body_plausible": "PENDING",
                "left_hand_plausible": "PENDING",
                "right_hand_plausible": "PENDING",
                "temporal_continuity": "PENDING",
                "catastrophic_failure": "PENDING",
                "reviewer": "",
                "review_notes": "",
            }
        )
        if len(records) == samples:
            break
    if len(records) != samples:
        raise ValueError(
            f"Only {len(records)} unique source groups are available; requested {samples}"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "queue": str(output.resolve()),
        "queue_sha256": sha256_file(output),
        "samples": samples,
        "unique_source_groups": len(records),
        "seed": seed,
        "decision": "PENDING until every review field is completed by a named reviewer",
    }
    report_path = output.with_suffix(".json")
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build_queue(args.manifest.resolve(), args.output.resolve(), args.samples, args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
