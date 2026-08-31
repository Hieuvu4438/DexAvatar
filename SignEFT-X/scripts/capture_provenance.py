#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.provenance import capture_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(capture_provenance(
        args.out.resolve(), args.repo.resolve(), args.lock.resolve(), args.config.resolve(),
        args.manifest.resolve(), args.evaluator.resolve(), set(args.require),
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

