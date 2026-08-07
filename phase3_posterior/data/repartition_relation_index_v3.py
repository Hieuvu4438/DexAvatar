"""Repartition immutable R2 sidecars under the expanded signer protocol."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from phase3_posterior.data.cache_schema import SCHEMA_VERSION, load_index
from phase3_posterior.data.prepare_how2sign_signers_v2 import SPLIT_SIGNERS
from phase3_posterior.provenance import atomic_json, sha256_file


def _signer_split(signer: str) -> str:
    value = int(signer.rsplit("_", 1)[-1])
    matches = [split for split, signers in SPLIT_SIGNERS.items() if value in signers]
    if len(matches) != 1:
        raise ValueError(f"How2Sign signer has no unique v2 split: {signer}")
    return matches[0]


def build(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    entries = []
    source_hashes = {}
    for split in ("train", "val", "calibration"):
        manifest = args.input_root.resolve() / "splits" / f"{split}.json"
        source_hashes[str(manifest)] = sha256_file(manifest)
        for entry in load_index(manifest):
            destination = _signer_split(entry.signer) if entry.source == "how2sign" else split
            if destination == "calibration" and entry.source != "how2sign":
                # No generic calibration source exists in the v2 cache today.
                continue
            entries.append(replace(entry, split=destination))
    test_manifest = args.test_root.resolve() / "splits" / "test.json"
    source_hashes[str(test_manifest)] = sha256_file(test_manifest)
    entries.extend(load_index(test_manifest))

    by_split = defaultdict(list)
    signer_splits: dict[tuple[str, str], str] = {}
    group_splits: dict[tuple[str, str], str] = {}
    clip_ids: set[tuple[str, str]] = set()
    for entry in entries:
        identity = (entry.source, entry.clip_id)
        if identity in clip_ids:
            raise ValueError(f"Duplicate clip identity: {identity}")
        clip_ids.add(identity)
        for key, table, label in (
            ((entry.source, entry.signer), signer_splits, "signer"),
            ((entry.source, entry.source_group), group_splits, "source group"),
        ):
            previous = table.setdefault(key, entry.split)
            if previous != entry.split:
                raise ValueError(f"{label} leakage {key}: {previous}/{entry.split}")
        if not Path(entry.relation_path).is_file():
            raise FileNotFoundError(entry.relation_path)
        by_split[entry.split].append(entry)

    output.mkdir(parents=True)
    hashes = {}
    counts = {}
    for split, values in sorted(by_split.items()):
        target = output / "splits" / f"{split}.json"
        atomic_json(
            target,
            {
                "schema_version": SCHEMA_VERSION,
                "clips": [entry.__dict__ for entry in values],
            },
        )
        hashes[split] = sha256_file(target)
        counts[split] = len(values)
    expected = {"train", "val", "test"}
    if set(by_split) != expected:
        raise ValueError(f"Expected exactly {sorted(expected)}, got {sorted(by_split)}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "expanded-how2sign-signer-components-with-sealed-signer10-v3",
        "split_counts": counts,
        "split_sha256": hashes,
        "source_manifest_sha256": source_hashes,
        "source_signer_group_disjoint": True,
        "lane_l_reads": 0,
        "how2sign_split_signers": {
            split: sorted(signers) for split, signers in SPLIT_SIGNERS.items()
        },
    }
    atomic_json(output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
