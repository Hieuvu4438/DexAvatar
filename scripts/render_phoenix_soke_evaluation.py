#!/usr/bin/env python3
"""Render the locked PHOENIX/SOKE PA-MPJPE JSON as a concise report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "signal4d-phoenix-soke-pampjpe-v1"
VARIANT_LABELS = {
    "initializer": "Frozen H32 + WiLoR initializer",
    "transformer_always": "Transformer, correction always applied",
    "transformer_gated": "Transformer, dev-calibrated gate",
}


def _number(value: Any) -> float:
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"Non-finite evaluation value: {value}")
    return result


def render(payload: dict[str, Any]) -> str:
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected schema: {payload.get('schema')}")
    if payload.get("mode") != "final_test_evaluation":
        raise ValueError(f"Not a final-test evaluation: {payload.get('mode')}")
    metrics = payload["metrics"]
    reference = payload["soke_table3_reference"]
    body_ref = _number(reference["phoenix_body_pa_mpjpe_mm"])
    hand_ref = _number(reference["phoenix_hand_pa_mpjpe_mm"])
    rows = []
    for key, label in VARIANT_LABELS.items():
        values = metrics[key]
        body = _number(values["body_pa_mpjpe_mm"])
        hand = _number(values["hand_pa_mpjpe_mm"])
        rows.append(
            f"| {label} | {body:.4f} | {hand:.4f} | "
            f"{body - body_ref:+.4f} | {hand - hand_ref:+.4f} |"
        )
    rows.append(
        f"| SOKE Table 3 reference | {body_ref:.4f} | {hand_ref:.4f} | "
        "0.0000 | 0.0000 |"
    )
    comparison = payload["comparison"]
    verdict = (
        "BEATS SOKE ON BOTH REPORTED REGIONS"
        if bool(comparison["beats_soke_both"])
        else "DOES NOT BEAT SOKE ON BOTH REPORTED REGIONS"
    )
    metric_definition = payload["metric_definition"]
    lines = [
        "# PHOENIX-2014T reconstruction evaluation",
        "",
        f"**Verdict: {verdict}.**",
        "",
        f"Official test: **{int(payload['clips'])} clips / "
        f"{int(payload['frames']):,} frames**. Lower is better; all values are mm.",
        "",
        "| Method | Body PA-MPJPE | Hand PA-MPJPE | Body Δ vs SOKE | Hand Δ vs SOKE |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        "## Region verdict",
        "",
        f"- Body beats SOKE: **{bool(comparison['beats_soke_body'])}**.",
        f"- Hand beats SOKE: **{bool(comparison['beats_soke_hand'])}**.",
        f"- Both beat SOKE: **{bool(comparison['beats_soke_both'])}**.",
        "",
        "## Metric contract",
        "",
        f"- Body: {metric_definition['body']}.",
        f"- Hands: {metric_definition['hand']}.",
        f"- Aggregation: {metric_definition['aggregation']}.",
        f"- Decoder: {metric_definition['decoder']}.",
        "",
        "## Comparability boundary",
        "",
        payload["protocol_difference"],
        "",
        "This is a metric-compatible comparison, not an identical training task: "
        "SOKE reports tokenizer encode/decode reconstruction after joint multilingual "
        "training, whereas this run refines a target-independent RGB expert initializer "
        "and trains only on official PHOENIX train.",
        "",
        "## Frozen lineage",
        "",
        f"- Test manifest: `{payload['manifest']}`",
        f"- Test manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Checkpoint SHA-256: `{payload['checkpoint_sha256']}`",
        f"- Dev calibration: `{payload['calibration']}`",
        f"- Dev calibration SHA-256: `{payload['calibration_sha256']}`",
        f"- Config: `{payload['config']}`",
        f"- Config SHA-256: `{payload['config_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def _write_report(output: Path, report: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(report)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = render(payload)
    _write_report(output, report)
    print(output)


if __name__ == "__main__":
    main()
