#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.export.diagnostics import render_front_side_points
from signpk.geometry.topology import load_obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Render front/side mesh diagnostics")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    parser.add_argument(
        "--canonical-camera-y-down",
        action="store_true",
        help="input is internal +y-down geometry rather than benchmark-exported y-up OBJ",
    )
    args = parser.parse_args()
    meshes = sorted(args.prediction_root.glob("*/meshes/*.obj"))
    if not meshes:
        raise FileNotFoundError(f"no exported meshes below {args.prediction_root}")
    for index, mesh in enumerate(meshes):
        if index % args.stride:
            continue
        vertices, _ = load_obj(mesh)
        sign = mesh.parents[1].name
        render_front_side_points(
            vertices,
            args.output_root / sign / f"{mesh.stem}.png",
            title=f"{sign} / {mesh.stem}",
            y_up=not args.canonical_camera_y_down,
        )
    print(f"rendered diagnostics to {args.output_root}")


if __name__ == "__main__":
    main()
