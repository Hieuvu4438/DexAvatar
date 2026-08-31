from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from signpccx.evaluation.parse_metrics import parse_global_metrics
from signpccx.io import atomic_write_json, atomic_write_text, sha256_file


OFFICIAL_EVALUATOR_SHA256 = "2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300"


def run_official_evaluator(
    evaluator: Path,
    evaluate_folder: Path,
    gt_folder: Path,
    signs_file: Path,
    segments_file: Path,
    output: Path,
    method: str,
    python: str = sys.executable,
) -> dict[str, object]:
    digest = sha256_file(evaluator)
    if digest != OFFICIAL_EVALUATOR_SHA256:
        raise RuntimeError(f"Official evaluator checksum mismatch: {digest}")
    command = [
        python,
        str(evaluator),
        "--method",
        method,
        "--central",
        "true",
        "--evaluate_folder",
        str(evaluate_folder),
        "--gt_folder",
        str(gt_folder),
        "--sign_file",
        str(signs_file),
        "--sign_seg",
        str(segments_file),
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "official_stdout.txt", result.stdout)
    atomic_write_text(output / "official_stderr.txt", result.stderr)
    record: dict[str, object] = {
        "schema_version": "signpccx.official-evaluation.v1",
        "command": command,
        "evaluator_sha256": digest,
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        atomic_write_json(output / "official_result.json", record)
        raise RuntimeError(f"Official evaluator failed ({result.returncode}); see {output}")
    record["metrics_mm"] = parse_global_metrics(result.stdout + "\n" + result.stderr)
    atomic_write_json(output / "official_result.json", record)
    return record

