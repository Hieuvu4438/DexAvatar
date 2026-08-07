"""Create immutable Phase 3 How2Sign manifests with signer-disjoint splits."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from phase2_refiner.data.cache_schema import load_cache_clip
from phase3_posterior.provenance import atomic_json, sha256_file


SPLIT_SIGNERS = {
    "train": frozenset({3, 5, 8}),
    "val": frozenset({1, 2}),
    "calibration": frozenset({4, 9, 11}),
}


def _paths(manifest: Path) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    values = payload.get("clips", payload)
    return [
        (manifest.parent / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
        for value in values
    ]


def _signer(source_clip: str) -> int:
    match = re.search(r"-(\d+)-rgb_front$", source_clip)
    if match is None:
        raise ValueError(f"Cannot parse How2Sign signer from {source_clip!r}")
    return int(match.group(1))


def prepare(input_manifests: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    clips: dict[str, tuple[Path, dict, str]] = {}
    for manifest in input_manifests:
        original_split = manifest.stem
        for path in _paths(manifest.resolve()):
            clip = load_cache_clip(path)
            metadata = json.loads(clip.metadata_json)
            previous = clips.setdefault(clip.clip_id, (path, metadata, original_split))
            if previous[0] != path:
                raise ValueError(
                    f"Duplicate clip_id with different paths: {clip.clip_id}"
                )

    by_split: dict[str, list[str]] = defaultdict(list)
    metadata_rows = []
    group_splits: dict[str, str] = {}
    signer_splits: dict[int, str] = {}
    for clip_id, (path, metadata, original_split) in sorted(clips.items()):
        signer = _signer(str(metadata.get("source_clip", "")))
        matches = [name for name, values in SPLIT_SIGNERS.items() if signer in values]
        if len(matches) != 1:
            raise ValueError(f"Signer {signer} has no unique Phase 3 split")
        split = matches[0]
        group = str(metadata.get("source_group", metadata.get("video_id", clip_id)))
        previous_group_split = group_splits.setdefault(group, split)
        if previous_group_split != split:
            raise ValueError(
                f"How2Sign source group {group} bridges {previous_group_split}/{split}"
            )
        previous_signer_split = signer_splits.setdefault(signer, split)
        if previous_signer_split != split:
            raise AssertionError((signer, previous_signer_split, split))
        by_split[split].append(str(path))
        metadata_rows.append(
            {
                "clip_id": clip_id,
                "signer": f"how2sign_signer_{signer:02d}",
                "source_group": group,
                "source_clip": metadata.get("source_clip"),
                "original_split": original_split,
                "phase3_split": split,
                "clip_sha256": sha256_file(path),
            }
        )

    output.mkdir(parents=True)
    split_hashes = {}
    for split in SPLIT_SIGNERS:
        target = output / f"{split}.json"
        atomic_json(
            target,
            {
                "schema": "phase3-how2sign-signer-split-v1",
                "split": split,
                "signers": sorted(SPLIT_SIGNERS[split]),
                "clips": by_split[split],
            },
        )
        split_hashes[split] = sha256_file(target)
    metadata_path = output / "signer_metadata.json"
    atomic_json(
        metadata_path,
        {
            "schema": "phase3-how2sign-signer-metadata-v1",
            "resolver": "terminal integer before -rgb_front",
            "records": metadata_rows,
        },
    )
    report = {
        "passed": True,
        "clips": len(metadata_rows),
        "split_counts": {key: len(by_split[key]) for key in SPLIT_SIGNERS},
        "split_signers": {key: sorted(values) for key, values in SPLIT_SIGNERS.items()},
        "source_groups": len(group_splits),
        "source_group_disjoint": True,
        "signer_disjoint": True,
        "metadata_sha256": sha256_file(metadata_path),
        "split_sha256": split_hashes,
    }
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
