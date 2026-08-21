from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from ..config import load_method_config
from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact
from ..optimization.smplx_solver import fit_smplx_sequence
from ..utils.hashing import sha256_file
from ..utils.seed import seed_everything


def run(
    config_path: str,
    manifest_path: str,
    cache_root: str,
    output_root: str,
    model_path: str,
    device: str = "cuda",
    warm_start_root: str | None = None,
) -> None:
    config = load_method_config(config_path)
    seed_everything(config.seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    clip_runtime_seconds: dict[str, float] = {}
    cache_hashes: dict[str, str] = {}
    warm_start_hashes: dict[str, str] = {}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
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
        warm_start = None
        if warm_start_root is not None:
            warm_start, warm_metadata = PredictionArtifact.load(
                Path(warm_start_root) / item.clip_id
            )
            warm_start_hashes[item.clip_id] = str(warm_metadata["artifact_sha256"])
        prediction, diagnostics = fit_smplx_sequence(
            batch,
            metadata,
            config,
            model_path,
            item.fps,
            str(output_root / "logs" / f"{item.clip_id}.jsonl"),
            warm_start,
        )
        prediction.save(
            output_root / "predictions" / item.clip_id,
            {
                "schema_version": "1.0",
                "clip_id": item.clip_id,
                "coordinate_convention": metadata["camera_convention"],
                "length_unit": metadata["length_unit"],
                "status": "success",
                "method_name": config.method_name,
                "config_sha256": config.sha256,
                "smplx_model_sha256": diagnostics["model_sha256"],
            },
        )
        diagnostics_path = output_root / "predictions" / item.clip_id / "factor_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        clip_runtime_seconds[item.clip_id] = time.perf_counter() - clip_started
        cache_hashes[item.clip_id] = str(metadata["artifact_sha256"])
        if device.startswith("cuda"):
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
        "cache_hashes": cache_hashes,
        "warm_start_root": (
            str(Path(warm_start_root).resolve()) if warm_start_root is not None else None
        ),
        "warm_start_hashes": warm_start_hashes,
        "smplx_model_sha256": sha256_file(model_path),
        "seed": config.seed,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runtime_seconds": time.perf_counter() - started,
        "clip_runtime_seconds": clip_runtime_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "frames": sum(len(item.frame_ids) for item in manifest),
        "clips": len(manifest),
    }
    (output_root / "run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
