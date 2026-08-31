#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.evaluation_audited import paired_sign_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    print(json.dumps(paired_sign_bootstrap(
        args.candidate.resolve(), args.baseline.resolve(), args.out.resolve(),
        replicates=args.replicates, seed=args.seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

