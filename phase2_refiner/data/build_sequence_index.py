"""Build explicit, auditable train/validation/test cache split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from phase2_refiner.data.cache_schema import load_cache_clip


VALID_SPLITS = {"train", "val", "test", "calibration"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_indices(cache_root: Path, assignments: Path, output: Path) -> dict:
    """Require explicit source/signer assignments; never guess scientific splits."""
    cache_paths = sorted((cache_root / "clips").glob("*.npz"))
    if not cache_paths:
        raise ValueError(f"No caches under {cache_root / 'clips'}")
    with assignments.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "split", "source", "signer"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Assignments CSV requires columns: {sorted(required)}")
    by_clip = {row["clip_id"]: row for row in rows}
    if len(by_clip) != len(rows):
        raise ValueError("Assignments contain duplicate clip_id values")

    split_entries: dict[str, list[str]] = defaultdict(list)
    split_sources: dict[str, set[str]] = defaultdict(set)
    split_signers: dict[str, set[str]] = defaultdict(set)
    audit = []
    for path in cache_paths:
        clip = load_cache_clip(path)
        if clip.clip_id not in by_clip:
            raise ValueError(f"No split assignment for {clip.clip_id}")
        row = by_clip[clip.clip_id]
        split = row["split"].strip().lower()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r} for {clip.clip_id}")
        source = row["source"].strip()
        signer = row["signer"].strip()
        if not source or not signer:
            raise ValueError(f"Missing source/signer for {clip.clip_id}")
        split_sources[split].add(source)
        split_signers[split].add(signer)
        relative = str(Path("..") / path.relative_to(cache_root))
        split_entries[split].append(relative)
        audit.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(clip.frame_names),
                "source": source,
                "signer": signer,
                "split": split,
                "sha256": _sha256(path),
            }
        )

    for left in VALID_SPLITS:
        for right in VALID_SPLITS:
            if left >= right:
                continue
            signer_overlap = split_signers[left] & split_signers[right]
            source_overlap = split_sources[left] & split_sources[right]
            if signer_overlap or source_overlap:
                raise ValueError(
                    f"Non-disjoint {left}/{right}: signers={sorted(signer_overlap)}, "
                    f"sources={sorted(source_overlap)}"
                )

    output.mkdir(parents=True, exist_ok=False)
    for split in sorted(VALID_SPLITS):
        payload = {
            "split": split,
            "clips": sorted(split_entries[split]),
            "assignments": str(assignments.resolve()),
            "assignments_sha256": _sha256(assignments),
        }
        with (output / f"{split}.json").open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    with (output / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {split: len(entries) for split, entries in split_entries.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_indices(
        args.cache_root.resolve(), args.assignments.resolve(), args.output.resolve()
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
