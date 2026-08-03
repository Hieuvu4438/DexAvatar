"""Fail-closed P3-G0 cache, leakage, license, and manual-review audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_posterior.data.cache_schema import load_index
from phase3_posterior.provenance import atomic_json, sha256_file


def audit(index_paths: list[Path], manual_quality: Path) -> dict:
    entries = []
    for path in index_paths:
        entries.extend(load_index(path))
    group_splits: dict[tuple[str, str], str] = {}
    signer_splits: dict[tuple[str, str], str] = {}
    leakage = []
    hash_failures = []
    relation_failures = []
    unverified_licenses = set()
    unknown_signers = []
    for entry in entries:
        group_identity = (entry.source, entry.source_group)
        previous = group_splits.setdefault(group_identity, entry.split)
        if previous != entry.split:
            leakage.append(
                {"source_group": group_identity, "splits": [previous, entry.split]}
            )
        if entry.signer == "unknown":
            unknown_signers.append(entry.clip_id)
        else:
            signer_identity = (entry.source, entry.signer)
            signer_split = signer_splits.setdefault(signer_identity, entry.split)
            if signer_split != entry.split:
                leakage.append(
                    {
                        "signer": signer_identity,
                        "splits": [signer_split, entry.split],
                    }
                )
        if sha256_file(entry.clip_path) != entry.clip_sha256:
            hash_failures.append(entry.clip_path)
        if not entry.relation_path or not Path(entry.relation_path).is_file():
            relation_failures.append(entry.clip_id)
        elif sha256_file(entry.relation_path) != entry.relation_sha256:
            relation_failures.append(entry.relation_path)
        if "REVIEW_REQUIRED" in entry.license_id or not entry.license_id:
            unverified_licenses.add(entry.license_id)
    with manual_quality.open("r", encoding="utf-8") as handle:
        manual = json.load(handle)
    reviewed = int(manual.get("reviewed_clips", 0))
    catastrophic = int(manual.get("catastrophic_failures", reviewed))
    failure_rate = catastrophic / max(reviewed, 1)
    checks = {
        "no_leakage": not leakage,
        "disjoint": not leakage and not unknown_signers,
        "hashes_licenses": not hash_failures and not unverified_licenses,
        "complete_relations": not relation_failures,
        "manual_review_at_least_300": reviewed >= 300,
        "manual_failure_below_10pct": reviewed >= 300 and failure_rate < 0.10,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics_for_gate": {
            "no_leakage": checks["no_leakage"],
            "disjoint": checks["disjoint"],
            "hashes_licenses": checks["hashes_licenses"],
            "manual_failure_rate": failure_rate,
        },
        "counts": {"clips": len(entries), "reviewed_clips": reviewed},
        "diagnostics": {
            "leakage": leakage,
            "hash_failures": hash_failures,
            "relation_failures": relation_failures,
            "unverified_licenses": sorted(unverified_licenses),
            "unknown_signer_clips": unknown_signers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--manual-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = audit(args.index, args.manual_quality)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
