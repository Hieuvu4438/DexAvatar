#!/usr/bin/env python3
"""Factorial ablation of shared-beta refinement and post-beta pose refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from signeft.canonical.refinement import (
    PARAMETER_SHAPES,
    _stack_parameters,
    canonical_refit,
    initializer_frame_paths,
)
from signeft.io.obj import write_obj
from signeft.io_utils import atomic_write_json, load_config, sha256_file
from signeft.manifest import read_jsonl


BOUNDARY_X180 = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def path(config: dict, key: str) -> Path:
    return Path(config["paths"][key]).resolve()


def identity_variant(baseline: Path, output: Path, beta_key: str) -> Path:
    source = baseline / "identity" / "signer.npz"
    destination = output / "identity" / f"{beta_key}.npz"
    if destination.is_file():
        return destination
    with np.load(source, allow_pickle=False) as archive:
        beta = np.asarray(archive[beta_key], dtype=np.float32).reshape(10)
        robust = np.asarray(archive["robust_beta"], dtype=np.float32).reshape(10)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, beta=beta, robust_beta=robust)
    atomic_write_json(destination.with_suffix(".json"), {
        "schema_version": "signeft.canonical-component-identity.v1",
        "selected_beta": beta_key,
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "beta": beta.tolist(),
    })
    return destination


def direct_decode(config: dict, baseline: Path, output: Path, beta_key: str,
                  batch_size: int) -> None:
    import smplx

    root = output / f"beta_{beta_key}__pose_without" / "meshes"
    summary = root.parent / "summary.json"
    if summary.is_file():
        return
    with np.load(baseline / "identity" / "signer.npz", allow_pickle=False) as archive:
        beta_np = np.asarray(archive[beta_key], dtype=np.float32).reshape(1, 10)
    device = str(config["runtime"]["device"])
    model = smplx.create(
        str(path(config, "smplx_model_root")), model_type="smplx", gender="neutral",
        num_betas=10, use_pca=False, use_face_contour=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    faces = np.asarray(model.faces, dtype=np.int64)
    manifests = sorted((baseline / "manifests").glob("*.jsonl"))
    count = 0
    started = time.perf_counter()
    for sign_index, manifest in enumerate(manifests, 1):
        records = read_jsonl(manifest)
        frames = [initializer_frame_paths(path(config, "initializer_root"), r) for r in records]
        arrays = _stack_parameters(frames)
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            destinations = [root / r.sign / f"{r.source_frame_id:06d}.obj" for r in records[start:end]]
            if all(item.is_file() for item in destinations):
                count += len(destinations)
                continue
            kwargs = {
                key: torch.as_tensor(arrays[key][start:end], dtype=torch.float32, device=device)
                for key in PARAMETER_SHAPES if key != "betas"
            }
            kwargs["betas"] = torch.as_tensor(beta_np, dtype=torch.float32, device=device).expand(end-start, -1)
            with torch.no_grad():
                vertices = model(return_verts=True, **kwargs).vertices.cpu().numpy() * BOUNDARY_X180
            for destination, value in zip(destinations, vertices, strict=True):
                if not destination.is_file():
                    write_obj(destination, value.astype(np.float32), faces)
                count += 1
        print(f"direct {beta_key}: {sign_index}/{len(manifests)} {manifest.stem}", flush=True)
    atomic_write_json(summary, {
        "schema_version": "signeft.canonical-component-ablation.v1",
        "beta_refinement": beta_key == "beta",
        "post_beta_pose_refit": False,
        "frames": count,
        "signs": len(manifests),
        "wall_seconds": time.perf_counter() - started,
        "objective_uses_ground_truth": False,
    })


def refit_robust(config: dict, baseline: Path, output: Path,
                 partition_index: int | None = None,
                 partition_count: int | None = None) -> None:
    identity = identity_variant(baseline, output, "robust_beta")
    fit_root = output / "beta_robust_beta__pose_with" / "canonical_fit"
    selected_signs = None
    if partition_index is not None:
        if partition_count is None or not 0 <= partition_index < partition_count:
            raise ValueError("invalid partition")
        names = [item.stem for item in sorted((baseline / "manifests").glob("*.jsonl"))]
        selected_signs = set(names[partition_index::partition_count])
    if not (fit_root / "run_manifest.json").is_file():
        settings = config["canonicalization"]
        canonical_refit(
            path(config, "initializer_root"), baseline / "manifests", identity,
            path(config, "smplx_model_root"), path(config, "mano_smplx_ids"), fit_root,
            device=str(config["runtime"]["device"]),
            steps=int(settings["steps"]), learning_rate=float(settings["learning_rate"]),
            chunk_size=int(settings["chunk_size"]), hand_weight=float(settings["hand_weight"]),
            whole_mesh_weight=float(settings["whole_mesh_weight"]),
            pose_anchor_weight=float(settings["pose_anchor_weight"]),
            max_hand_residual_mm=float(settings["max_hand_residual_mm"]),
            signs=selected_signs,
        )
    if selected_signs is not None:
        return
    mesh_root = fit_root.parent / "meshes"
    for manifest in sorted((baseline / "manifests").glob("*.jsonl")):
        records = read_jsonl(manifest)
        with np.load(fit_root / "clips" / manifest.stem / "mesh_parametric_final.npz", allow_pickle=False) as archive:
            vertices = np.asarray(archive["mesh_parametric"], dtype=np.float32)
            faces = np.asarray(archive["faces"], dtype=np.int64)
            frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        if list(frame_ids) != [r.source_frame_id for r in records]:
            raise RuntimeError(f"frame mismatch: {manifest.stem}")
        for record, value in zip(records, vertices, strict=True):
            destination = mesh_root / record.sign / f"{record.source_frame_id:06d}.obj"
            if not destination.is_file():
                write_obj(destination, value, faces)
    atomic_write_json(fit_root.parent / "summary.json", {
        "schema_version": "signeft.canonical-component-ablation.v1",
        "beta_refinement": False,
        "post_beta_pose_refit": True,
        "frames": sum(1 for _ in mesh_root.glob("*/*.obj")),
        "signs": len(list(mesh_root.glob("*"))),
        "objective_uses_ground_truth": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("direct-robust", "direct-optimized", "refit-robust"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/full1493"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ablation_canonical_components"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--partition-index", type=int)
    parser.add_argument("--partition-count", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    baseline, output = args.baseline_root.resolve(), args.output_root.resolve()
    if args.mode == "direct-robust":
        direct_decode(config, baseline, output, "robust_beta", args.batch_size)
    elif args.mode == "direct-optimized":
        direct_decode(config, baseline, output, "beta", args.batch_size)
    else:
        refit_robust(
            config, baseline, output, args.partition_index, args.partition_count
        )


if __name__ == "__main__":
    main()
