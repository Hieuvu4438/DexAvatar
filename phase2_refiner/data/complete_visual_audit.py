"""Complete a rendered Phase-2 visual audit with explicit reviewer provenance.

This command never edits the source review queue.  It verifies that every queue
row has a corresponding rendered evidence image, records one decision per clip,
and writes both a completed CSV and a machine-readable gate report.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from phase2_refiner.provenance import sha256_file


REVIEW_COLUMNS = (
    "body_plausible",
    "left_hand_plausible",
    "right_hand_plausible",
    "temporal_continuity",
    "catastrophic_failure",
    "reviewer",
    "review_notes",
)


def complete_audit(
    queue: Path,
    evidence_manifest: Path,
    output_csv: Path,
    output_report: Path,
    reviewer: str,
    modality: str,
    catastrophic_indices: set[int],
) -> dict:
    if not reviewer.strip():
        raise ValueError("A named --reviewer is required")
    if output_csv.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite an existing audit artifact")

    with queue.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    if not rows or columns is None:
        raise ValueError("Audit queue is empty")
    missing = set(REVIEW_COLUMNS).difference(columns)
    if missing:
        raise ValueError(f"Queue is missing review columns: {sorted(missing)}")

    with evidence_manifest.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    if evidence.get("queue_sha256") != sha256_file(queue):
        raise ValueError("Evidence was not rendered from this exact queue")
    images = evidence.get("clip_images", [])
    if len(images) != len(rows):
        raise ValueError(
            f"Evidence/queue length mismatch: {len(images)} images, {len(rows)} rows"
        )
    missing_images = [path for path in images if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} evidence images")
    invalid_indices = catastrophic_indices.difference(range(1, len(rows) + 1))
    if invalid_indices:
        raise ValueError(f"Invalid one-based clip indices: {sorted(invalid_indices)}")

    default_note = (
        f"{modality}; four uniformly sampled frames; source video with projected "
        "H32 mesh, independent How2Sign 2D tracks, and teacher side view. "
        "This screen detects catastrophic pseudo-target failure; it does not "
        "certify millimetre-level pose or fine-finger accuracy."
    )
    completed = []
    for index, row in enumerate(rows, start=1):
        failed = index in catastrophic_indices
        result = dict(row)
        result.update(
            {
                "body_plausible": "FAIL" if failed else "PASS",
                "left_hand_plausible": "FAIL" if failed else "PASS",
                "right_hand_plausible": "FAIL" if failed else "PASS",
                "temporal_continuity": "FAIL" if failed else "PASS",
                "catastrophic_failure": "YES" if failed else "NO",
                "reviewer": reviewer.strip(),
                "review_notes": default_note,
            }
        )
        completed.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(completed)

    failures = len(catastrophic_indices)
    threshold = 0.10
    failure_rate = failures / len(rows)
    report = {
        "schema_version": 1,
        "audit_date": date.today().isoformat(),
        "reviewer": reviewer.strip(),
        "review_modality": modality,
        "queue": str(queue.resolve()),
        "queue_sha256": sha256_file(queue),
        "evidence_manifest": str(evidence_manifest.resolve()),
        "evidence_manifest_sha256": sha256_file(evidence_manifest),
        "completed_csv": str(output_csv.resolve()),
        "completed_csv_sha256": sha256_file(output_csv),
        "clips_reviewed": len(rows),
        "distinct_source_groups": len({row["source_group"] for row in rows}),
        "frames_inspected_per_clip": evidence.get("frames_per_clip"),
        "catastrophic_failures": failures,
        "catastrophic_failure_rate": failure_rate,
        "required_failure_rate_below": threshold,
        "gate_pass": failure_rate < threshold,
        "scope_limit": (
            "Catastrophic pseudo-target screening only; fine-finger and "
            "millimetre-level accuracy require quantitative validation."
        ),
    }
    with output_report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--modality",
        default="AI visual review by Codex",
        help="Disclosed review modality recorded in every row and the report",
    )
    parser.add_argument(
        "--catastrophic-index",
        type=int,
        action="append",
        default=[],
        help="One-based queue index judged catastrophic; repeat as needed",
    )
    args = parser.parse_args()
    report = complete_audit(
        args.queue.resolve(),
        args.evidence_manifest.resolve(),
        args.output_csv.resolve(),
        args.output_report.resolve(),
        args.reviewer,
        args.modality,
        set(args.catastrophic_index),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
