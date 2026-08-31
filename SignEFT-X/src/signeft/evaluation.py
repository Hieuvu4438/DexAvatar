from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

from signeft.io_utils import atomic_write_json, atomic_write_text, sha256_file


_METRIC = re.compile(
    r"\[(?P<method>[^\]]+)\]:\s+(?P<metric>[^:]+):\s+(?P<value>[0-9.+-eE]+)\s+\(mm\)"
)


def parse_metrics(text: str) -> dict[str, float]:
    metrics = {
        " ".join(match.group("metric").lower().split()): float(match.group("value"))
        for match in _METRIC.finditer(text)
    }
    required = {
        "tr all", "tr above pelvis upper body", "tr above pelvis minus face",
        "tr above pelvis minus head", "tr left hand", "tr right hand",
    }
    if missing := required - set(metrics):
        raise ValueError(f"missing official metrics: {sorted(missing)}")
    return metrics


def evaluate_official(
    evaluator: Path,
    expected_sha256: str,
    prediction_root: Path,
    gt_root: Path,
    signs_file: Path,
    segments_file: Path,
    output_root: Path,
    method: str,
    python: str = sys.executable,
) -> dict[str, object]:
    digest = sha256_file(evaluator)
    if digest != expected_sha256:
        raise RuntimeError(f"official evaluator checksum mismatch: {digest}")
    command = [
        python, str(evaluator), "--method", method, "--central", "true",
        "--evaluate_folder", str(prediction_root), "--gt_folder", str(gt_root),
        "--sign_file", str(signs_file), "--sign_seg", str(segments_file),
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_root / "official_stdout.txt", result.stdout)
    atomic_write_text(output_root / "official_stderr.txt", result.stderr)
    report: dict[str, object] = {
        "schema_version": "signeft.official-evaluation.v1",
        "command": command,
        "evaluator_sha256": digest,
        "returncode": result.returncode,
    }
    if result.returncode == 0:
        report["metrics_mm"] = parse_metrics(result.stdout + "\n" + result.stderr)
    atomic_write_json(output_root / "official_result.json", report)
    if result.returncode != 0:
        raise RuntimeError(f"official evaluator failed: see {output_root}")
    return report
