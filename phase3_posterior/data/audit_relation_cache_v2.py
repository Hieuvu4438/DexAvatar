"""Fail-closed audit for corrected R2 continuous relation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase3_posterior.data.cache_schema import load_index, load_relation_sidecar
from phase3_posterior.provenance import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    blockers = []
    groups: dict[tuple[str, str], str] = {}
    signers: dict[tuple[str, str], str] = {}
    clips = 0
    target_values = 0
    independent_how2sign = 0
    contact_positive = 0
    contact_total = 0
    for split in ("train", "val", "calibration"):
        manifest = args.root / "splits" / f"{split}.json"
        for entry in load_index(manifest):
            clips += 1
            for identity, table, label in (
                ((entry.source, entry.source_group), groups, "source-group"),
                ((entry.source, entry.signer), signers, "signer"),
            ):
                previous = table.setdefault(identity, split)
                if previous != split:
                    blockers.append(
                        f"{label} leakage for {identity}: {previous} versus {split}"
                    )
            if sha256_file(entry.relation_path) != entry.relation_sha256:
                blockers.append(f"relation hash mismatch: {entry.clip_id}")
                continue
            relation = load_relation_sidecar(entry.relation_path)
            if relation.target_edge_features is None:
                blockers.append(f"missing target edge features: {entry.clip_id}")
                continue
            target_values += int(relation.target_edge_features.size)
            if not np.isfinite(relation.target_edge_features).all():
                blockers.append(f"non-finite target edge features: {entry.clip_id}")
            metadata = json.loads(relation.metadata_json)
            if entry.source == "how2sign":
                independent_how2sign += int(
                    metadata.get("target_geometry_provider")
                    == "smplx_decoded_independent_target_v2"
                )
                source_node, target_node = relation.edge_index
                masks = (source_node >= 10) ^ (target_node >= 10)
                valid = relation.contact_valid[:, masks]
                target = relation.contact_target[:, masks]
                contact_positive += int((target & valid).sum())
                contact_total += int(valid.sum())
    expected_how2sign = sum(
        entry.source == "how2sign"
        for split in ("train", "val", "calibration")
        for entry in load_index(args.root / "splits" / f"{split}.json")
    )
    if independent_how2sign != expected_how2sign:
        blockers.append(
            "How2Sign independent target provider coverage is incomplete: "
            f"{independent_how2sign}/{expected_how2sign}"
        )
    if contact_positive == 0 or contact_total == 0:
        blockers.append("How2Sign hand-body contact audit has no valid positives")
    result = {
        "schema_version": 1,
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "clips": clips,
        "target_feature_values": target_values,
        "how2sign_independent_target_clips": independent_how2sign,
        "how2sign_expected_clips": expected_how2sign,
        "how2sign_hand_body_contact_positive": contact_positive,
        "how2sign_hand_body_contact_valid": contact_total,
        "how2sign_hand_body_contact_rate": contact_positive
        / max(contact_total, 1),
        "manifest_sha256": sha256_file(args.root / "manifest.json"),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(2)
if __name__ == "__main__":
    main()
