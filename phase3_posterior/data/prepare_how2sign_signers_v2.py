"""Build the expanded signer-disjoint How2Sign split protocol for P3-G2.

The old v1 manifests remain immutable.  V2 uses six official-development
signers for fitting, keeps signers 01 and 02 separate for validation and
calibration, and reserves official-test signer 10 as a sealed final test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phase2_refiner.data.cache_schema import load_cache_clip
from phase3_posterior.data.prepare_how2sign_signers import _paths, _signer
from phase3_posterior.provenance import atomic_json, sha256_file


SPLIT_SIGNERS = {
    "train": frozenset({3, 4, 5, 8, 9, 11}),
    "val": frozenset({1, 2}),
    "test": frozenset({10}),
}


def prepare(input_manifests: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    clips: dict[str, tuple[Path, dict, str]] = {}
    for manifest in input_manifests:
        for path in _paths(manifest.resolve()):
            clip = load_cache_clip(path)
            metadata = json.loads(clip.metadata_json)
            previous = clips.setdefault(clip.clip_id, (path, metadata, manifest.stem))
            if previous[0] != path:
                raise ValueError(f"Duplicate clip_id with different paths: {clip.clip_id}")

    by_split: dict[str, list[str]] = defaultdict(list)
    rows = []
    signer_splits: dict[int, str] = {}
    group_splits: dict[str, str] = {}
    official_test_signers: set[int] = set()
    for clip_id, (path, metadata, original_split) in sorted(clips.items()):
        signer = _signer(str(metadata.get("source_clip", "")))
        matches = [name for name, values in SPLIT_SIGNERS.items() if signer in values]
        if len(matches) != 1:
            raise ValueError(f"Signer {signer} has no unique v2 split")
        split = matches[0]
        official_split = str(metadata.get("official_split", original_split))
        if split == "test":
            if official_split != "test":
                raise ValueError(f"Test signer clip is not official test: {path}")
            official_test_signers.add(signer)
        group = str(metadata.get("source_group", clip_id))
        previous_group = group_splits.setdefault(group, split)
        if previous_group != split:
            raise ValueError(
                f"How2Sign source group {group} bridges {previous_group}/{split}"
            )
        previous_signer = signer_splits.setdefault(signer, split)
        if previous_signer != split:
            raise AssertionError((signer, previous_signer, split))
        by_split[split].append(str(path))
        rows.append(
            {
                "clip_id": clip_id,
                "clip_path": str(path),
                "clip_sha256": sha256_file(path),
                "signer": f"how2sign_signer_{signer:02d}",
                "source_group": group,
                "official_split": official_split,
                "phase3_split": split,
            }
        )
    missing = set().union(*SPLIT_SIGNERS.values()) - set(signer_splits)
    if missing:
        raise ValueError(f"Required signer identities are absent: {sorted(missing)}")
    if official_test_signers != {10}:
        raise ValueError(f"Expected only official-test signer 10: {official_test_signers}")

    output.mkdir(parents=True)
    split_hashes = {}
    for split, signers in SPLIT_SIGNERS.items():
        if not by_split[split]:
            raise ValueError(f"Empty v2 split: {split}")
        target = output / f"{split}.json"
        atomic_json(
            target,
            {
                "schema": "phase3-how2sign-signer-split-v2",
                "split": split,
                "signers": sorted(signers),
                "clips": by_split[split],
            },
        )
        split_hashes[split] = sha256_file(target)
    metadata_path = output / "signer_metadata.json"
    atomic_json(
        metadata_path,
        {
            "schema": "phase3-how2sign-signer-metadata-v2",
            "resolver": "terminal integer before -rgb_front",
            "records": rows,
        },
    )
    report = {
        "passed": True,
        "protocol": "six-fit-two-val-sealed-official-test-v2",
        "clips": len(rows),
        "split_counts": {key: len(by_split[key]) for key in SPLIT_SIGNERS},
        "split_signers": {key: sorted(value) for key, value in SPLIT_SIGNERS.items()},
        "source_groups": len(group_splits),
        "source_group_disjoint": True,
        "signer_disjoint": True,
        "official_test_signer_10_sealed": True,
        "metadata_sha256": sha256_file(metadata_path),
        "split_sha256": split_hashes,
    }
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
