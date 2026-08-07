"""Finalize a visually reviewed Phase 3 geometry audit without hiding decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase3_posterior.provenance import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--existing-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--failed-clip", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    pending = json.loads(args.pending.read_text(encoding="utf-8"))
    failed = set(args.failed_clip)
    records = []
    if args.existing_csv is not None:
        with args.existing_csv.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                catastrophic = str(row["catastrophic_failure"]).upper() == "YES"
                records.append(
                    {
                        "clip_id": row["clip_id"],
                        "source": "how2sign",
                        "decision": "FAIL" if catastrophic else "PASS",
                        "reviewer": row["reviewer"],
                        "review_notes": row["review_notes"],
                        "evidence_source": str(args.existing_csv.resolve()),
                    }
                )
    for row in pending["records"]:
        item = dict(row)
        evidence = Path(item["evidence_image"])
        relation = Path(item["relation_path"])
        if sha256_file(relation) != item["relation_sha256"]:
            raise ValueError(f"Relation changed after audit rendering: {relation}")
        item["evidence_image_sha256"] = sha256_file(evidence)
        item["decision"] = "FAIL" if item["clip_id"] in failed else "PASS"
        item["reviewer"] = args.reviewer
        item["review_notes"] = (
            "Catastrophic geometry/contact evidence failure."
            if item["clip_id"] in failed
            else "Four-frame relation geometry sheet visually checked: finite torso, wrists, hands, and relational edges; no catastrophic failure."
        )
        records.append(item)
    unique = {row["clip_id"] for row in records}
    if len(unique) != len(records):
        raise ValueError("Visual audit contains duplicate clip IDs")
    failures = sum(row["decision"] == "FAIL" for row in records)
    report = {
        "schema": "phase3-manual-quality-contact-audit-v1",
        "reviewed_clips": len(records),
        "catastrophic_failures": failures,
        "catastrophic_failure_rate": failures / max(len(records), 1),
        "reviewer": args.reviewer,
        "existing_csv": str(args.existing_csv.resolve()) if args.existing_csv else None,
        "existing_csv_sha256": sha256_file(args.existing_csv)
        if args.existing_csv
        else None,
        "relation_audit": str(args.pending.resolve()),
        "relation_audit_sha256": sha256_file(args.pending),
        "records": records,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
