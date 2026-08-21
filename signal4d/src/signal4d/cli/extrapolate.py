from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..geometry.so3 import slerp
from ..io.predictions import PredictionArtifact
from ..models.smplx_wrapper import SMPLXWrapper
from ..optimization.smplx_solver import _make_state
from ..utils.hashing import sha256_file


def extrapolate_rotations(
    baseline: torch.Tensor, candidate: torch.Tensor, alpha: float
) -> torch.Tensor:
    if baseline.shape != candidate.shape or baseline.shape[-2:] != (3, 3):
        raise ValueError("rotation hypotheses must have matching [...,3,3] shapes")
    return slerp(baseline, candidate, alpha)


def run(
    manifest_path: str,
    candidate_root: str,
    baseline_root: str,
    cache_root: str,
    model_path: str,
    alpha: float,
    output_root: str,
    device: str = "cuda",
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 4.0:
        raise ValueError("extrapolation alpha must be in [0,4]")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    model = SMPLXWrapper(model_path).to(device)
    clips = load_manifest(manifest_path)
    started = time.perf_counter()
    source_hashes: dict[str, str] = {}
    for item in clips:
        candidate_dir = Path(candidate_root) / item.clip_id
        baseline_dir = Path(baseline_root) / item.clip_id
        cache_dir = Path(cache_root) / item.clip_id
        candidate, candidate_meta = PredictionArtifact.load(candidate_dir)
        baseline, baseline_meta = PredictionArtifact.load(baseline_dir)
        observations, cache_meta = ObservationBatch.load(cache_dir)
        observations.validate_against(item)
        if (
            candidate.frame_ids.tolist() != item.frame_ids
            or baseline.frame_ids.tolist() != item.frame_ids
        ):
            raise ValueError(f"extrapolation frame mismatch for {item.clip_id}")
        if candidate.rotations is None or baseline.rotations is None:
            raise ValueError("extrapolation requires complete rotations")
        diagnostics_path = candidate_dir / "factor_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        shape_key = (
            "legacy_betas_mean"
            if diagnostics.get("shape_source") == "legacy_biomech"
            else "betas_mean"
        )
        if cache_meta.get(shape_key) is None:
            raise ValueError(f"missing {shape_key} for {item.clip_id}")
        rotations = extrapolate_rotations(
            baseline.rotations.to(device), candidate.rotations.to(device), alpha
        )
        stored_translation = torch.lerp(
            baseline.translation.to(device), candidate.translation.to(device), alpha
        )
        internal_translation = stored_translation * stored_translation.new_tensor(
            [1.0, -1.0, -1.0]
        )
        betas = torch.tensor(
            cache_meta[shape_key], dtype=rotations.dtype, device=device
        )[None]
        state = _make_state(rotations, internal_translation, betas, trainable=False)
        with torch.inference_mode():
            decoded = model(state)
        prediction = PredictionArtifact(
            frame_ids=candidate.frame_ids,
            joints_3d=decoded.joints[:, :55],
            rotations=rotations,
            translation=stored_translation,
            vertices=decoded.vertices,
            risk_score=candidate.risk_score,
            abstain=candidate.abstain,
            uncertainty=candidate.uncertainty,
            contact_probability=candidate.contact_probability,
            contacts=candidate.contacts,
        )
        prediction.save(
            output / "predictions" / item.clip_id,
            {
                "schema_version": "1.0",
                "method_name": "signal4d_m1_geodesic_extrapolation",
                "clip_id": item.clip_id,
                "coordinate_convention": candidate_meta.get("coordinate_convention"),
                "length_unit": candidate_meta.get("length_unit"),
                "smplx_model_sha256": model.model_hash,
                "candidate_artifact_sha256": candidate_meta["artifact_sha256"],
                "baseline_artifact_sha256": baseline_meta["artifact_sha256"],
                "alpha": alpha,
                "shape_source": diagnostics.get("shape_source"),
                "gt_used": False,
            },
        )
        diagnostics.update(
            {
                "geodesic_extrapolation_alpha": alpha,
                "geodesic_extrapolation_gt_used": False,
            }
        )
        result_diagnostics = output / "predictions" / item.clip_id / "factor_diagnostics.json"
        result_diagnostics.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in (
            candidate_dir / "prediction.safetensors",
            baseline_dir / "prediction.safetensors",
            cache_dir / "observations.safetensors",
            diagnostics_path,
        ):
            source_hashes[str(path)] = sha256_file(path)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "schema_version": "1.0",
        "method_name": "signal4d_m1_geodesic_extrapolation",
        "alpha": alpha,
        "clips": len(clips),
        "frames": sum(len(item.frame_ids) for item in clips),
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": model.model_hash,
        "source_hashes": source_hashes,
        "gt_used": False,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
