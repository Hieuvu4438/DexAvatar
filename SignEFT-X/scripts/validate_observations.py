#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.observations.validate import validate_nlf_cache, validate_pose_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("pose", "nlf"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    function = validate_pose_cache if args.kind == "pose" else validate_nlf_cache
    print(json.dumps(
        function(args.manifest.resolve(), args.root.resolve(), args.out.resolve()),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()

