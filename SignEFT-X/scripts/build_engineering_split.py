#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.data.split import build_engineering_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        build_engineering_split(args.manifest.resolve(), args.out.resolve()),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()

