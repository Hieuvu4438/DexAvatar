"""Strict Phase 3 regional evaluation through the stable common evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2_refiner.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("data/evaluation_from_author/data/data"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = evaluate(
        args.manifest,
        args.prediction,
        args.output,
        Path(".").resolve(),
        args.baseline,
        args.assets_root,
        args.bootstrap_samples,
        args.seed,
        False,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
