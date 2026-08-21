from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..data.cache import ObservationBatch
from ..data.manifest import ClipManifest, write_manifest
from ..utils.seed import seed_everything


def _ground_truth(frames: int, joints: int, clip_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.linspace(0, 1, frames, dtype=torch.float64)
    base = torch.linspace(-0.25, 0.25, joints, dtype=torch.float64)
    xyz = torch.zeros((frames, joints, 3), dtype=torch.float64)
    xyz[..., 0] = base[None] + 0.025 * torch.sin(2 * torch.pi * (time[:, None] + base[None]))
    xyz[..., 1] = 0.12 * torch.cos(torch.pi * time[:, None]) + 0.02 * base[None]
    xyz[..., 2] = 1.2 + 0.05 * torch.sin(2 * torch.pi * time[:, None])
    # Fast meaningful articulation around the midpoint.
    transition = torch.sigmoid((time - 0.5) * 80)
    xyz[:, -4:, 0] += (0.04 + 0.005 * clip_index) * transition[:, None]
    midpoint = joints // 2
    # A short contact event with clean onset/offset.
    active = (time >= 0.35) & (time <= 0.65)
    average = 0.5 * (xyz[:, midpoint - 1] + xyz[:, midpoint])
    xyz[active, midpoint - 1] = average[active] - torch.tensor([0.003, 0.0, 0.0])
    xyz[active, midpoint] = average[active] + torch.tensor([0.003, 0.0, 0.0])
    contacts = torch.zeros((frames, 2), dtype=torch.bool)
    contacts[active, 0] = True
    return xyz, contacts


def create_synthetic_artifact(
    output_root: str | Path,
    num_clips: int = 3,
    frames: int = 24,
    joints: int = 12,
    sources: int = 3,
    seed: int = 12345,
) -> Path:
    seed_everything(seed)
    root = Path(output_root)
    cache_root = root / "cache"
    gt_root = root / "ground_truth"
    rows: list[ClipManifest] = []
    for clip_index in range(num_clips):
        clip_id = f"synthetic_{clip_index:03d}"
        truth, contacts = _ground_truth(frames, joints, clip_index)
        observations = truth[:, None].expand(frames, sources, joints, 3).clone()
        noise_scale = torch.tensor([0.004, 0.009, 0.018], dtype=truth.dtype)[:sources]
        noise = torch.randn_like(observations) * noise_scale[None, :, None, None]
        observations += noise
        if sources > 2:
            observations[:, 2, -4:, 0] += 0.025  # controlled biased estimator
        valid = torch.ones((frames, sources, joints), dtype=torch.bool)
        valid[7:11, 0, -4:] = False
        valid[14:18, min(1, sources - 1), :3] = False
        observations = torch.where(valid[..., None], observations, torch.zeros_like(observations))
        confidence = torch.where(
            valid, 1 - noise_scale[None, :, None] / 0.03, torch.zeros_like(valid, dtype=truth.dtype)
        )
        ensemble_center = observations.sum(1) / valid.sum(1).clamp_min(1)[..., None]
        disagreement = torch.linalg.vector_norm(observations - ensemble_center[:, None], dim=-1)
        features = torch.stack(
            (
                1 - confidence,
                disagreement,
                (~valid).to(truth.dtype),
                torch.linspace(0, 1, frames, dtype=truth.dtype)[:, None, None].expand_as(
                    confidence
                ),
            ),
            dim=-1,
        )
        batch = ObservationBatch(
            frame_ids=torch.arange(frames, dtype=torch.int64),
            joints_3d=observations,
            valid_3d=valid,
            features=features,
        )
        batch.save(
            cache_root / clip_id,
            {
                "schema_version": "1.0",
                "clip_id": clip_id,
                "sources": [
                    {"source_id": index, "name": f"synthetic_source_{index}"}
                    for index in range(sources)
                ],
                "camera_convention": "opencv_x_right_y_down_z_forward",
                "length_unit": "meter",
            },
        )
        (gt_root / clip_id).mkdir(parents=True, exist_ok=True)
        save_file(
            {"joints_3d": truth, "contacts": contacts},
            gt_root / clip_id / "ground_truth.safetensors",
        )
        (gt_root / clip_id / "metadata.json").write_text(
            json.dumps(
                {
                    "source": "deterministic synthetic oracle",
                    "seed": seed,
                    "clip_index": clip_index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rows.append(
            ClipManifest(
                dataset="synthetic",
                clip_id=clip_id,
                signer_id=f"synthetic_signer_{clip_index % 2}",
                split="development",
                fps=25.0,
                frame_ids=list(range(frames)),
                image_relpaths=[f"synthetic/{clip_id}/{index:05d}.png" for index in range(frames)],
                frame_start=0,
                frame_end_exclusive=frames,
                is_contiguous=True,
                gt_relpath=f"ground_truth/{clip_id}/ground_truth.safetensors",
                allowed_for_hparam_selection=True,
            )
        )
    manifest_path = root / "manifest.jsonl"
    write_manifest(rows, manifest_path)
    return manifest_path
