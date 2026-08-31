"""Audit independent double-annotation agreement; this alone does not close G1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dcg_sign4d.contact.agreement import AgreementThresholds, audit_double_annotations
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--minimum-events", type=int, required=True)
    parser.add_argument("--minimum-edge-agreement", type=float, required=True)
    parser.add_argument("--minimum-mean-interval-iou", type=float, required=True)
    parser.add_argument("--maximum-mean-boundary-error-frames", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.annotations)
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    thresholds = AgreementThresholds(
        minimum_events=args.minimum_events,
        minimum_edge_agreement=args.minimum_edge_agreement,
        minimum_mean_interval_iou=args.minimum_mean_interval_iou,
        maximum_mean_boundary_error_frames=args.maximum_mean_boundary_error_frames,
    )
    report = audit_double_annotations(records, thresholds)
    report.update(
        {
            "scientific_scope": "ANNOTATION_AGREEMENT_ONLY_G1_ALSO_REQUIRES_PSEUDO_LABEL_AUDIT",
            "development_only": args.development_only,
            "annotation_sha256": file_sha256(source),
            "audit_config_sha256": canonical_hash(thresholds),
        }
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable label audit exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".agreement_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    (output / "agreement_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    marker = (
        "ANNOTATION_AGREEMENT_PASS"
        if report["agreement_gate_pass"]
        else "ANNOTATION_AGREEMENT_FAILED"
    )
    os.replace(incomplete, output / marker)
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report["agreement_gate_pass"]:
        raise RuntimeError("contact annotation agreement gate failed")


if __name__ == "__main__":
    main()
