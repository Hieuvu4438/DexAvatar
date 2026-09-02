#!/usr/bin/env python3
"""Resume a disjoint canonicalization partition for multi-process execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from signeft.canonical.refinement import canonical_refit
from signeft.io_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = {name: Path(value) for name, value in config["paths"].items()}
    output = paths["output_root"]
    manifests = output / "manifests"
    names = [path.stem for path in sorted(manifests.glob("*.jsonl"))]
    selected = set(names[args.skip + args.offset :: args.stride])
    settings = config["canonicalization"]
    result = canonical_refit(
        paths["initializer_root"],
        manifests,
        output / "identity/signer.npz",
        paths["smplx_model_root"],
        paths["mano_smplx_ids"],
        output / "canonical_fit",
        device=str(config["runtime"]["device"]),
        steps=int(settings["steps"]),
        learning_rate=float(settings["learning_rate"]),
        chunk_size=int(settings["chunk_size"]),
        hand_weight=float(settings["hand_weight"]),
        whole_mesh_weight=float(settings["whole_mesh_weight"]),
        pose_anchor_weight=float(settings["pose_anchor_weight"]),
        max_hand_residual_mm=float(settings["max_hand_residual_mm"]),
        signs=selected,
    )
    print(f"partition complete: {result['signs']} signs / {result['frames']} frames")


if __name__ == "__main__":
    main()
