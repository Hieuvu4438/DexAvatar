"""Render Phase 3 result PKLs using the stable Phase 2 renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase2_refiner.render import render_result_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    total = 0
    for result_dir in sorted(args.prediction.glob("*/smplifyx/results")):
        mesh_dir = result_dir.parent / "meshes"
        total += render_result_directory(
            result_dir, mesh_dir, args.model_folder, device=args.device, overwrite=False
        )
    print(f"Rendered {total} Phase 3 meshes")


if __name__ == "__main__":
    main()
