#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.observations.heatmaps import export_sapiens_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sapiens-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = export_sapiens_heatmaps(
        args.manifest.resolve(), args.out.resolve(), args.checkpoint.resolve(),
        args.sapiens_root.resolve(), device=args.device, batch_size=args.batch_size,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
