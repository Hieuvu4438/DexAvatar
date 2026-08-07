"""Fail-closed P3-G0 cache, leakage, license, and manual-review audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phase3_posterior.data.cache_schema import load_index, load_relation_sidecar
from phase3_posterior.provenance import atomic_json, sha256_file


def audit(
    index_paths: list[Path], manual_quality: Path, license_evidence: Path | None = None
) -> dict:
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
    coverage_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    coverage_count: dict[str, int] = defaultdict(int)
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
        else:
            relation = load_relation_sidecar(entry.relation_path)
            metadata = json.loads(relation.metadata_json)
            coverage_sum[entry.source]["keypoint_3d"] += float(
                relation.node_valid.mean()
            )
            coverage_sum[entry.source]["torso"] += float(
                relation.node_valid[:, :10].mean()
            )
            coverage_sum[entry.source]["wrist_hand"] += float(
                relation.node_valid[:, 10:].mean()
            )
            coverage_sum[entry.source]["relation_edge"] += float(
                relation.edge_valid.mean()
            )
            coverage_sum[entry.source]["contact_target"] += float(
                relation.contact_valid.mean()
            )
            valid_contact = relation.contact_valid
            coverage_sum[entry.source]["contact_positive"] += float(
                relation.contact_target[valid_contact].mean()
                if valid_contact.any()
                else 0.0
            )
            coverage_count[entry.source] += 1
            provider = str(metadata.get("input_geometry_provider", ""))
            if not (
                provider.startswith("smplx_decoded_")
                or provider == "interhand_official_joint_3d_v1"
            ):
                relation_failures.append(f"non-decoded:{entry.relation_path}")
        if "REVIEW_REQUIRED" in entry.license_id or not entry.license_id:
            unverified_licenses.add(entry.license_id)
    with manual_quality.open("r", encoding="utf-8") as handle:
        manual = json.load(handle)
    reviewed = int(manual.get("reviewed_clips", manual.get("clips_reviewed", 0)))
    catastrophic = int(manual.get("catastrophic_failures", reviewed))
    failure_rate = catastrophic / max(reviewed, 1)
    reviewed_records = manual.get("records", [])
    record_ids = {
        str(item.get("clip_id"))
        for item in reviewed_records
        if isinstance(item, dict) and item.get("clip_id")
    }
    manual_hash_failures = []
    for item in reviewed_records:
        if not isinstance(item, dict):
            manual_hash_failures.append("non-mapping-record")
            continue
        for path_key, hash_key in (
            ("relation_path", "relation_sha256"),
            ("evidence_image", "evidence_image_sha256"),
        ):
            value = item.get(path_key)
            expected = item.get(hash_key)
            if value is None:
                continue
            path = Path(str(value))
            if not path.is_file() or sha256_file(path) != expected:
                manual_hash_failures.append(str(value))
    manual_evidence_complete = (
        reviewed >= 300 and len(record_ids) >= 300 and not manual_hash_failures
    )
    license_evidence_ok = False
    license_evidence_sha256 = None
    if license_evidence is not None and license_evidence.is_file():
        license_payload = json.loads(license_evidence.read_text(encoding="utf-8"))
        recorded = {
            str(item.get("license_id")) for item in license_payload.get("records", [])
        }
        used = {entry.license_id for entry in entries}
        license_evidence_ok = used <= recorded
        license_evidence_sha256 = sha256_file(license_evidence)
    coverage = {
        source: {
            key: value / coverage_count[source] for key, value in sorted(values.items())
        }
        for source, values in sorted(coverage_sum.items())
    }
    coverage_requirements = {
        "keypoint_3d": 0.70,
        "torso": 0.20,
        "wrist_hand": 0.99,
        "relation_edge": 0.20,
        "contact_target": 0.20,
        "contact_positive": 1e-6,
    }
    relation_coverage_ok = bool(coverage) and all(
        all(
            values.get(key, 0.0) + 1e-9 >= threshold
            for key, threshold in coverage_requirements.items()
        )
        for values in coverage.values()
    )
    checks = {
        "no_leakage": not leakage,
        "disjoint": not leakage and not unknown_signers,
        "hashes_licenses": (
            not hash_failures and not unverified_licenses and license_evidence_ok
        ),
        "complete_relations": not relation_failures,
        "relation_coverage_meets_source_contract": relation_coverage_ok,
        "manual_review_at_least_300": manual_evidence_complete,
        "manual_failure_below_10pct": reviewed >= 300 and failure_rate < 0.10,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "passed": all(checks.values()),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "checks": checks,
        "metrics_for_gate": {
            "no_leakage": checks["no_leakage"],
            "disjoint": checks["disjoint"],
            "hashes_licenses": checks["hashes_licenses"],
            "cache_audit_passed": all(checks.values()),
            "manual_failure_rate": failure_rate,
        },
        "counts": {"clips": len(entries), "reviewed_clips": reviewed},
        "relation_coverage": coverage,
        "relation_coverage_requirements": coverage_requirements,
        "license_evidence": {
            "path": str(license_evidence.resolve()) if license_evidence else None,
            "sha256": license_evidence_sha256,
            "passed": license_evidence_ok,
        },
        "diagnostics": {
            "leakage": leakage,
            "hash_failures": hash_failures,
            "relation_failures": relation_failures,
            "unverified_licenses": sorted(unverified_licenses),
            "unknown_signer_clips": unknown_signers,
            "manual_unique_record_ids": len(record_ids),
            "manual_hash_failures": manual_hash_failures,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--manual-quality", type=Path, required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = audit(args.index, args.manual_quality, args.license_evidence)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
