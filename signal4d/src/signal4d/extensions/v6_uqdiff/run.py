from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from ...config import load_method_config
from ...data.cache import ObservationBatch
from ...data.manifest import load_manifest
from ...io.predictions import PredictionArtifact
from ...models.smplx_wrapper import SMPLXWrapper
from ...utils.hashing import sha256_file
from ...utils.seed import seed_everything
from .config import load_v6_config
from .dposer_bridge import DPoserXBridge
from .refiner import refine_v5_clip


def run(
    config_path: str,
    manifest_path: str,
    cache_root: str,
    output_root: str,
    model_path: str,
    device_name: str = "cuda",
    selected_clips: set[str] | None = None,
) -> None:
    config = load_v6_config(config_path)
    base_method = load_method_config(config.base_method_config)
    seed_everything(config.dposer.noise_seed)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V6 run: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".run_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    manifest = load_manifest(manifest_path)
    if selected_clips is not None:
        known = {item.clip_id for item in manifest}
        if not selected_clips <= known:
            raise ValueError(f"unknown requested clips: {sorted(selected_clips - known)}")
        manifest = [item for item in manifest if item.clip_id in selected_clips]
    device = torch.device(device_name)
    model = SMPLXWrapper(model_path).to(device).eval()
    model.requires_grad_(False)
    dposer = DPoserXBridge(config.dposer, device) if config.dposer.enabled else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    runtime: dict[str, float] = {}
    parent_hashes: dict[str, str] = {}
    accepted_frames = 0
    for item in manifest:
        clip_started = time.perf_counter()
        batch, metadata = ObservationBatch.load(Path(cache_root) / item.clip_id)
        batch.validate_against(item)
        batch = ObservationBatch(
            **{
                name: value.to(device)
                for name, value in batch.__dict__.items()
                if value is not None
            }
        )
        base, base_metadata = PredictionArtifact.load(
            Path(config.warm_start_root) / item.clip_id
        )
        base_values = {
            name: value.to(device) if value is not None else None
            for name, value in base.__dict__.items()
        }
        base = PredictionArtifact(**base_values)
        result = refine_v5_clip(
            batch,
            metadata,
            base,
            base_method,
            config,
            model,
            dposer,
            item.fps,
        )
        clip_output = output / "predictions" / item.clip_id
        result.prediction.save(
            clip_output,
            {
                "schema_version": "1.0",
                "status": "success",
                "clip_id": item.clip_id,
                "coordinate_convention": metadata["camera_convention"],
                "length_unit": metadata["length_unit"],
                "method_name": config.method_name,
                "config_sha256": config.sha256,
                "smplx_model_sha256": model.model_hash,
                "parent_v5_artifact_sha256": base_metadata["artifact_sha256"],
                "gt_used": False,
            },
        )
        diagnostics_text = json.dumps(result.diagnostics, indent=2, sort_keys=True) + "\n"
        for filename in ("v6_diagnostics.json", "factor_diagnostics.json"):
            (clip_output / filename).write_text(diagnostics_text, encoding="utf-8")
        accepted_frames += int(result.diagnostics["accepted_frames"])
        runtime[item.clip_id] = time.perf_counter() - clip_started
        parent_hashes[item.clip_id] = str(base_metadata["artifact_sha256"])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    run_metadata = {
        "schema_version": "1.0",
        "status": "success",
        "method_name": config.method_name,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config.sha256,
        "config_file_sha256": sha256_file(config_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "cache_root": str(Path(cache_root).resolve()),
        "warm_start_root": str(Path(config.warm_start_root).resolve()),
        "parent_v5_artifact_hashes": parent_hashes,
        "smplx_model_sha256": model.model_hash,
        "dposer_registry_sha256": sha256_file(config.dposer.checkpoint_registry),
        "gt_used": False,
        "clips": len(manifest),
        "frames": sum(len(item.frame_ids) for item in manifest),
        "accepted_frames": accepted_frames,
        "runtime_seconds": time.perf_counter() - started,
        "clip_runtime_seconds": runtime,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (output / "run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(prog="signal4d-v6-refine")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip", action="append")
    args = parser.parse_args()
    run(
        args.config,
        args.manifest,
        args.cache_root,
        args.output_root,
        args.model_path,
        args.device,
        set(args.clip) if args.clip else None,
    )


if __name__ == "__main__":
    main()
