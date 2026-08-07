"""Resumably prefill one source's relation sidecars before index finalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase2_refiner.render import create_smplx_model
from phase3_posterior.data.build_phase3_index import _manifest_paths
from phase3_posterior.data.build_relation_targets import (
    InterHandJointProvider,
    build_sidecar,
)
from phase3_posterior.data.cache_schema import save_relation_sidecar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = create_smplx_model(args.model_folder, device) if args.model_folder else None
    interhand_provider = InterHandJointProvider()
    seen = set()
    completed = skipped = 0
    for manifest in args.manifest:
        for clip_path in _manifest_paths(manifest.resolve()):
            if clip_path in seen:
                continue
            seen.add(clip_path)
            # Clip IDs are already unique by the Phase 2 schema.
            target = args.output_root / args.source_name / f"{clip_path.stem}.npz"
            if target.exists():
                skipped += 1
                continue
            save_relation_sidecar(
                target,
                build_sidecar(
                    clip_path,
                    model=model,
                    device=device,
                    interhand_provider=interhand_provider,
                ),
            )
            completed += 1
            if completed == 1 or completed % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "relation_prefill_progress",
                            "completed": completed,
                            "skipped": skipped,
                            "source": args.source_name,
                        }
                    ),
                    flush=True,
                )
    print(json.dumps({"completed": completed, "skipped": skipped}), flush=True)


if __name__ == "__main__":
    main()
