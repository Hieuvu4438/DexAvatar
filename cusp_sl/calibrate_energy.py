"""Fit robust energy normalizers from generated validation candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cusp_sl.selection import EnergyStatistics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    manifest_path = args.candidate_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_role") != "development_validation":
        raise ValueError("Energy statistics require development_validation candidates")
    if not all(item.get("energy_computed") is True for item in manifest["summaries"]):
        raise ValueError("Energy statistics require candidates with computed evidence")
    declared = {
        str(item["clip_id"]): str(item["prediction_sha256"])
        for item in manifest["summaries"]
    }
    paths = [
        args.candidate_root / "clips" / f"{clip_id}.npz"
        for clip_id in sorted(declared)
    ]
    terms = []
    for path in paths:
        clip_id = path.stem
        if sha256(path) != declared[clip_id]:
            raise ValueError(f"Candidate hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            terms.append(payload["energy_terms"])
    statistics = EnergyStatistics.fit(torch.from_numpy(np.concatenate(terms, axis=0)).float())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("cusp_sl_energy_statistics_v2"),
        role=np.asarray("development_candidate_energy_normalization"),
        median=statistics.median.numpy(),
        mad=statistics.mad.numpy(),
        source_files=np.asarray([str(path.resolve()) for path in paths]),
        source_candidate_manifest_sha256=np.asarray(sha256(manifest_path)),
        config_sha256=np.asarray(manifest["config_sha256"]),
        reliability_checkpoint_sha256=np.asarray(
            manifest["reliability_checkpoint_sha256"]
        ),
        flow_checkpoint_sha256=np.asarray(manifest["flow_checkpoint_sha256"]),
        gate_calibration_sha256=np.asarray(
            manifest.get("gate_calibration_sha256") or ""
        ),
    )
    print(json.dumps({"output": str(args.output), "clips": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
