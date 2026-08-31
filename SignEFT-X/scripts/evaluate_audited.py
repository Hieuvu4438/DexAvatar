#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.evaluation_audited import evaluate_audited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--official-result", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_audited(
        args.manifest.resolve(), args.protocol_lock.resolve(), args.predictions.resolve(),
        args.gt_root.resolve(), args.asset_root.resolve(), args.out.resolve(),
        official_result=args.official_result.resolve() if args.official_result else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

