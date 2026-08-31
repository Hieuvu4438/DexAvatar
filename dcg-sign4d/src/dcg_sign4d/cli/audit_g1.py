"""Composite G1 gate: frozen ontology plus gold agreement and pseudo-label quality."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement-report", required=True)
    parser.add_argument("--pseudo-label-report", required=True)
    parser.add_argument("--patch-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    agreement_path = Path(args.agreement_report)
    pseudo_path = Path(args.pseudo_label_report)
    patch_path = Path(args.patch_map)
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
    patch = PatchMap.load(patch_path)
    checks = {
        "annotation_agreement": agreement.get("agreement_gate_pass") is True,
        "pseudo_label_quality": pseudo.get("pseudo_label_gate_pass") is True,
        "patch_map_frozen": not patch.development_only and patch.scientific_status == "FROZEN",
        "agreement_not_development": agreement.get("development_only") is False,
        "pseudo_audit_not_development": pseudo.get("development_only") is False,
    }
    report = {
        "schema_version": "dcg_g1_gate_v1",
        "gate": "G1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "evidence_sha256": {
            "agreement_report": file_sha256(agreement_path),
            "pseudo_label_report": file_sha256(pseudo_path),
            "patch_map": file_sha256(patch_path),
        },
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable G1 audit exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".g1_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    (output / "G1_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, output / ("G1_PASS" if all(checks.values()) else "G1_BLOCKED"))
    print(json.dumps(report, sort_keys=True, indent=2))
    if not all(checks.values()):
        raise RuntimeError("G1 is blocked; development evidence cannot close the gate")


if __name__ == "__main__":
    main()
