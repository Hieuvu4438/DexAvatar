"""Build corrected R2 relation targets without mutating the Phase 3 v1 cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from phase2_refiner.render import create_smplx_model
from phase3_posterior.data.build_relation_targets import (
    InterHandJointProvider,
    build_sidecar,
)
from phase3_posterior.data.cache_schema import (
    SCHEMA_VERSION,
    load_index,
    save_relation_sidecar,
)
from phase3_posterior.provenance import atomic_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite corrected cache: {args.output}")
    if (args.output / "manifest.json").exists():
        raise FileExistsError(f"Corrected cache is already complete: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    model = create_smplx_model(args.model_folder, device)
    interhand = InterHandJointProvider()
    split_payloads: dict[str, list[dict]] = {}
    total = 0
    built = 0
    reused = 0
    for split in ("train", "val", "calibration"):
        source_manifest = args.input_root / "splits" / f"{split}.json"
        entries = load_index(source_manifest)
        result = []
        for entry in entries:
            target = args.output / "relations" / entry.source / f"{entry.clip_id}.npz"
            if target.exists():
                reused += 1
            else:
                save_relation_sidecar(
                    target,
                    build_sidecar(
                        entry.clip_path,
                        model=model,
                        device=device,
                        interhand_provider=interhand,
                    ),
                )
                built += 1
            updated = replace(
                entry,
                relation_path=str(target),
                relation_sha256=sha256_file(target),
            )
            updated.validate()
            result.append(updated.__dict__)
            total += 1
            if total == 1 or total % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "relation_v2_progress",
                            "processed": total,
                            "built": built,
                            "reused": reused,
                            "split": split,
                        }
                    ),
                    flush=True,
                )
        split_payloads[split] = result
    split_hashes = {}
    for split, clips in split_payloads.items():
        target = args.output / "splits" / f"{split}.json"
        atomic_json(target, {"schema_version": SCHEMA_VERSION, "clips": clips})
        split_hashes[split] = sha256_file(target)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "relation_schema_version": 2,
        "source_root": str(args.input_root.resolve()),
        "source_manifest_sha256": sha256_file(args.input_root / "manifest.json"),
        "model_folder": str(args.model_folder.resolve()),
        "splits": {key: len(value) for key, value in split_payloads.items()},
        "split_sha256": split_hashes,
        "built": built,
        "reused_after_resume": reused,
        "target_contract": "independent_smplx_target_v2",
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
