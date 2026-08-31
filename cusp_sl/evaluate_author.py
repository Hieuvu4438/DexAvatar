"""Run the immutable author evaluator while archiving command, hashes, and stdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from cusp_sl.config import load_config


METRIC_PATTERN = re.compile(r"\[[^\]]+\]: ([^:]+): ([0-9]+(?:\.[0-9]+)?) \(mm\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name", default="cusp_sl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    config = load_config(args.config)
    evaluator = Path("data/evaluation_from_author/evaluate_new_fitting.py")
    command = [
        sys.executable, str(evaluator), "--method", args.method_name,
        "--central", "true", "--evaluate_folder", str(args.prediction_root.resolve()),
        "--gt_folder", str(Path(config.protocol.gt_root).resolve()),
        "--sign_file", str(Path(config.protocol.signs_file).resolve()),
        "--sign_seg", str(Path(config.protocol.segments_file).resolve()),
    ]
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    args.output.mkdir(parents=True)
    (args.output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    metrics = {
        name.strip(): float(value)
        for name, value in METRIC_PATTERN.findall(completed.stdout)
    }
    required_metrics = {
        "Tr Above Pelvis Minus Face",
        "Tr Left Hand",
        "Tr Right Hand",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "command": command, "exit_code": completed.returncode,
        "evaluator_sha256": sha256(evaluator),
        "signs_sha256": sha256(Path(config.protocol.signs_file)),
        "segments_sha256": sha256(Path(config.protocol.segments_file)),
        "prediction_root": str(args.prediction_root.resolve()),
        "parsed_metrics": len(metrics),
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(completed.stdout)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    missing_metrics = required_metrics.difference(metrics)
    if missing_metrics:
        raise RuntimeError(
            f"Author evaluator omitted required metrics: {sorted(missing_metrics)}"
        )


if __name__ == "__main__":
    main()
