from __future__ import annotations

import argparse
import json

from dcg_sign4d.evaluation.compare import compare_per_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired clip-level evaluator comparison")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    result = compare_per_clip(
        args.baseline,
        args.candidate,
        metrics=tuple(args.metric),
        output_path=args.output,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
