"""Fit a frozen temperature calibrator from independent correctness labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dcg_sign4d.observations.calibration import TemperatureScaler
from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with logits [N,C] and labels [N]")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, choices=("calibration", "validation"))
    parser.add_argument("--label-protocol-id", required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--max-ece-regression", type=float, required=True)
    parser.add_argument(
        "--input-transform",
        choices=("precomputed_logits", "scalar_as_binary_logit"),
        default="precomputed_logits",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if args.label_protocol_id.strip().upper() in {"UNKNOWN", "TODO", "AUTHOR_REQUIRED"}:
        raise ValueError("label protocol must be frozen before calibration")
    if args.bins < 2 or args.max_iterations < 1 or args.max_ece_regression < 0:
        raise ValueError("invalid calibration/gate settings")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable calibration output exists: {output}")
    source = Path(args.input)
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "labels"}:
            raise ValueError("calibration NPZ must contain exactly logits and labels")
        logits = torch.from_numpy(np.asarray(arrays["logits"])).float()
        labels = torch.from_numpy(np.asarray(arrays["labels"])).long()
    if logits.ndim != 2 or logits.shape[1] < 2 or labels.shape != logits.shape[:1]:
        raise ValueError("calibration requires logits [N,C>=2] and labels [N]")
    if labels.numel() < 2 or torch.unique(labels).numel() < 2:
        raise ValueError("calibration labels require at least two represented classes")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("calibration label is outside the class topology")
    scaler = TemperatureScaler()
    report = scaler.fit(
        logits,
        labels,
        split=args.split,
        bins=args.bins,
        max_iter=args.max_iterations,
    )
    gate_pass = report.calibrated_ece <= report.raw_ece + args.max_ece_regression
    model_identity = {
        "method": "temperature_scaling",
        "temperature": report.temperature,
        "input_sha256": file_sha256(source),
        "manifest_sha256": file_sha256(args.manifest),
        "label_protocol_id": args.label_protocol_id,
        "fit_split": args.split,
        "bins": args.bins,
        "max_iterations": args.max_iterations,
        "development_only": args.development_only,
        "input_transform": args.input_transform,
    }
    payload = {
        "schema_version": "dcg_temperature_calibration_v1",
        "development_only": args.development_only,
        **report.__dict__,
        "gate_status": "PASS" if gate_pass else "FAIL",
        "gate_rule": {
            "metric": "ECE",
            "max_regression": args.max_ece_regression,
            "bins": args.bins,
            "max_iterations": args.max_iterations,
        },
        "class_support": torch.bincount(labels, minlength=logits.shape[1]).tolist(),
        "input_sha256": file_sha256(source),
        "manifest_sha256": file_sha256(args.manifest),
        "label_protocol_id": args.label_protocol_id,
        "calibration_model_sha256": canonical_hash(model_identity),
        "input_transform": args.input_transform,
    }
    output.mkdir(parents=True)
    incomplete = output / ".calibration_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    (output / "calibrator.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(
        incomplete,
        output / ("CALIBRATION_PASS" if gate_pass else "CALIBRATION_FAILED"),
    )
    print(json.dumps(payload, sort_keys=True, indent=2))
    if not gate_pass:
        raise RuntimeError("calibration failed the preregistered ECE gate")


if __name__ == "__main__":
    main()
