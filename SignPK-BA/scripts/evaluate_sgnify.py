#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.evaluation.trv2v_audited import AuditedTRV2VEvaluator, load_subsets
from signpk.utils.config import load_yaml, project_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict frame-identity SGNify TR-V2V evaluator")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/eval/trv2v.yaml")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--sign", action="append", help="repeat to evaluate a subset")
    parser.add_argument("--strict-frame-ids", action="store_true", help="required protocol flag")
    parser.add_argument("--include-class0-left", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not args.strict_frame_ids:
        raise ValueError("this evaluator requires --strict-frame-ids; ordinal pairing is forbidden")
    config = load_yaml(args.config)["evaluation"]
    evaluator = AuditedTRV2VEvaluator(
        load_subsets(project_path(config["data_root"], PROJECT_ROOT)),
        vertex_count=int(config.get("vertex_count", 10475)),
        class0_exclude_left_hand=bool(config.get("class0_exclude_left_hand", True))
        and not args.include_class0_left,
    )
    result = evaluator.evaluate_root(
        project_path(config["manifest_root"], PROJECT_ROOT),
        args.prediction_root.resolve(),
        signs=None if not args.sign else set(args.sign),
        official_evaluator_path=project_path(config["official_evaluator"], PROJECT_ROOT),
    )
    output = args.output_json or project_path(config["output_json"], PROJECT_ROOT)
    result.save(output)
    print("Region        mean(mm)  median(mm)  p95(mm)  frames")
    for name, summary in result.overall.items():
        values = [summary.mean_mm, summary.median_mm, summary.p95_mm]
        rendered = ["n/a" if value is None else f"{value:.4f}" for value in values]
        print(
            f"{name:<12} {rendered[0]:>9}  {rendered[1]:>10}  {rendered[2]:>7}  {summary.frames:>6}"
        )
    if result.subgroups:
        print(f"subgroups: {len(result.subgroups)} (see JSON)")
    print("protocol: audited_strict (official formula compatible; original script not executed)")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
