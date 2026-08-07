"""Create additive source-filtered Phase 3 manifests with disjointness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_posterior.data.cache_schema import SCHEMA_VERSION, load_index
from phase3_posterior.provenance import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-source", action="append", required=True)
    parser.add_argument("--split", action="append", default=["train", "val"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite filtered index: {args.output}")
    included = set(args.include_source)
    split_payloads = {}
    signer_splits: dict[tuple[str, str], str] = {}
    group_splits: dict[tuple[str, str], str] = {}
    split_hashes = {}
    for split in dict.fromkeys(args.split):
        source_manifest = args.input_root / "splits" / f"{split}.json"
        entries = [
            entry for entry in load_index(source_manifest) if entry.source in included
        ]
        if not entries:
            raise ValueError(f"Filtered {split} manifest is empty")
        for entry in entries:
            for identity, table, label in (
                ((entry.source, entry.signer), signer_splits, "signer"),
                ((entry.source, entry.source_group), group_splits, "source-group"),
            ):
                previous = table.setdefault(identity, split)
                if previous != split:
                    raise ValueError(
                        f"{label} leakage for {identity}: {previous} versus {split}"
                    )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "clips": [entry.__dict__ for entry in entries],
        }
        target = args.output / "splits" / f"{split}.json"
        atomic_json(target, payload)
        split_payloads[split] = len(entries)
        split_hashes[split] = sha256_file(target)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": "R2_geometry_only_R3_progression",
        "source_root": str(args.input_root.resolve()),
        "source_root_manifest_sha256": sha256_file(args.input_root / "manifest.json"),
        "included_sources": sorted(included),
        "split_counts": split_payloads,
        "split_sha256": split_hashes,
        "source_signer_group_disjoint": True,
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
