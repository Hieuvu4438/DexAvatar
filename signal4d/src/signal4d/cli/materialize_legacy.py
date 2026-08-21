from __future__ import annotations

import json
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..adapters.sgnify_local import _axis_angle_rotations
from ..data.manifest import load_manifest
from ..io.predictions import PredictionArtifact
from ..models.smplx_wrapper import SMPLXWrapper
from ..optimization.smplx_solver import _make_state
from ..utils.hashing import sha256_file


def _trusted_parameters(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    if not isinstance(value, dict):
        raise ValueError(f"legacy parameter file is not a mapping: {path}")
    return value


def _tensor(
    rows: list[dict[str, np.ndarray]], key: str, width: int, device: str
) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([np.asarray(row.get(key, np.zeros(width))).reshape(width) for row in rows])
    ).float().to(device)


def run(
    manifest_path: str,
    primary_root: str,
    primary_subpath: str,
    model_path: str,
    output_root: str,
    method_name: str,
    device: str = "cuda",
    fallback_root: str | None = None,
    fallback_subpath: str = "smplifyx/results",
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    primary = Path(primary_root)
    fallback = Path(fallback_root) if fallback_root else None
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    model = SMPLXWrapper(model_path).to(device)
    source_hashes: dict[str, str] = {}
    source_counts = {"primary": 0, "fallback": 0}
    started = time.perf_counter()
    for item in manifest:
        rows: list[dict[str, np.ndarray]] = []
        selected_sources: list[str] = []
        for frame_id in item.frame_ids:
            path = primary / item.clip_id / primary_subpath / f"low_{frame_id:03d}.pkl"
            source = "primary"
            if not path.is_file():
                if fallback is None:
                    raise FileNotFoundError(path)
                path = fallback / item.clip_id / fallback_subpath / f"low_{frame_id:03d}.pkl"
                source = "fallback"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(_trusted_parameters(path))
            selected_sources.append(source)
            source_counts[source] += 1
            source_hashes[str(path)] = sha256_file(path)
        rotations = torch.stack([_axis_angle_rotations(row) for row in rows]).to(device)
        translation = _tensor(rows, "transl", 3, device)
        betas = _tensor(rows, "betas", 10, device)
        state = _make_state(rotations, translation, betas, trainable=False)
        with torch.inference_mode():
            decoded = model(state)
        frames = len(item.frame_ids)
        uncertainty = torch.zeros((frames, 55), dtype=decoded.vertices.dtype, device=device)
        prediction = PredictionArtifact(
            frame_ids=torch.tensor(item.frame_ids, dtype=torch.int64, device=device),
            joints_3d=decoded.joints[:, :55],
            rotations=rotations,
            translation=translation * translation.new_tensor([1.0, -1.0, -1.0]),
            vertices=decoded.vertices,
            risk_score=torch.zeros((frames, 3), dtype=decoded.vertices.dtype, device=device),
            abstain=torch.zeros((frames, 3), dtype=torch.bool, device=device),
            uncertainty=uncertainty,
        )
        prediction.save(
            output / "predictions" / item.clip_id,
            {
                "schema_version": "1.0",
                "method_name": method_name,
                "clip_id": item.clip_id,
                "coordinate_convention": "opencv_x_right_y_down_z_forward",
                "length_unit": "meter",
                "smplx_model_sha256": model.model_hash,
                "source_selection": selected_sources,
                "primary_root": str(primary),
                "fallback_root": str(fallback) if fallback is not None else None,
                "legacy_parameter_decode": "per_frame_shape_pinned_smplx",
            },
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "schema_version": "1.0",
        "method_name": method_name,
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": model.model_hash,
        "clips": len(manifest),
        "frames": sum(len(item.frame_ids) for item in manifest),
        "source_counts": source_counts,
        "source_hashes": source_hashes,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
