from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..evaluation.sgnify import load_obj_vertices
from ..models.smplx_wrapper import SMPLXWrapper
from ..models.uncertainty import GroupCalibration, UncertaintyCalibrator, student_t_nll
from ..utils.hashing import sha256_file
from ..utils.seed import seed_everything


def _region(joint: int) -> str:
    if joint < 25:
        return "body"
    if joint < 40:
        return "left_hand"
    return "right_hand"


def run(
    manifest_path: str,
    cache_root: str,
    gt_root: str,
    model_path: str,
    output: str,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    seed: int = 12345,
    device: str = "cpu",
    conformal_clips: int = 4,
    sigma_min: float = 0.002,
    sigma_max: float = 0.5,
) -> dict[str, object]:
    seed_everything(seed)
    rows = load_manifest(manifest_path)
    if any(row.split != "calibration" or not row.allowed_for_calibration for row in rows):
        raise ValueError("uncertainty fitting accepts only an explicitly allowed calibration split")

    torch_device = torch.device(device)
    joint_regressor = SMPLXWrapper(model_path).model.J_regressor.detach().float().to(torch_device)
    records: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]] = {}
    cache_hashes: dict[str, str] = {}
    gt_hashes: dict[str, str] = {}
    gt_root_path = Path(gt_root)
    for row in rows:
        cache_dir = Path(cache_root) / row.clip_id
        batch, _ = ObservationBatch.load(cache_dir)
        batch.validate_against(row)
        batch = ObservationBatch(
            **{
                name: value.to(torch_device)
                for name, value in batch.__dict__.items()
                if value is not None
            }
        )
        target_vertices = []
        for frame_id in row.frame_ids:
            path = gt_root_path / row.clip_id / f"{frame_id * 2:05d}.obj"
            if not path.is_file():
                raise FileNotFoundError(path)
            target_vertices.append(load_obj_vertices(path))
            gt_hashes[str(path)] = sha256_file(path)
        target_joints = torch.einsum(
            "jv,tvc->tjc", joint_regressor, torch.stack(target_vertices).to(torch_device)
        )[:, :55]
        # Match the benchmark endpoint: estimate observation error after per-frame
        # translation alignment, using the body-source pelvis as the shared anchor.
        translation_delta = target_joints[:, 0] - batch.joints_3d[:, 0, 0]
        aligned_observations = batch.joints_3d + translation_delta[:, None, None]
        residual = torch.linalg.vector_norm(aligned_observations - target_joints[:, None], dim=-1)
        cache_hashes[row.clip_id] = sha256_file(cache_dir / "observations.safetensors")
        clip_groups: list[str] = []
        for _frame in range(batch.frame_ids.numel()):
            for source in range(batch.joints_3d.shape[1]):
                clip_groups.extend(
                    f"source_{source}:{_region(joint)}" for joint in range(batch.joints_3d.shape[2])
                )
        records[row.clip_id] = (batch.features, batch.valid_3d, residual, clip_groups)

    if conformal_clips < 1 or conformal_clips >= len(records):
        raise ValueError("conformal_clips must leave at least one clip for model fitting")
    ranked_ids = sorted(
        records,
        key=lambda clip: hashlib.sha256(f"{seed}:conformal:{clip}".encode()).hexdigest(),
    )
    conformal_ids = set(ranked_ids[-conformal_clips:])
    train_ids = [clip for clip in ranked_ids if clip not in conformal_ids]
    x = torch.cat([records[clip][0] for clip in train_ids])
    mask = torch.cat([records[clip][1] for clip in train_ids])
    target = torch.cat([records[clip][2] for clip in train_ids])
    model = UncertaintyCalibrator(x.shape[-1], sigma_min=sigma_min, sigma_max=sigma_max).to(
        torch_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, mask)["sigma_xyz"].mean(-1)
        loss = student_t_nll(target[mask], prediction[mask]).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        conformal_x = torch.cat([records[clip][0] for clip in sorted(conformal_ids)])
        conformal_mask = torch.cat([records[clip][1] for clip in sorted(conformal_ids)])
        conformal_target = torch.cat([records[clip][2] for clip in sorted(conformal_ids)])
        sigma = model(conformal_x, conformal_mask)["sigma_xyz"].mean(-1)
    conformal_groups = sum((records[clip][3] for clip in sorted(conformal_ids)), start=[])
    valid_groups = [
        group
        for group, keep in zip(conformal_groups, conformal_mask.flatten().tolist(), strict=True)
        if keep
    ]
    calibration = GroupCalibration.fit(
        conformal_target[conformal_mask].flatten(),
        sigma[conformal_mask].flatten(),
        valid_groups,
    )
    scores = conformal_target[conformal_mask].flatten() / sigma[conformal_mask].flatten()
    empirical_coverage_unclipped: dict[str, float] = {}
    empirical_coverage: dict[str, float] = {}
    for group, scale in calibration.scales.items():
        group_mask = torch.tensor([value == group for value in valid_groups], device=scores.device)
        empirical_coverage_unclipped[group] = float((scores[group_mask] <= scale).float().mean())
        group_radius = (sigma[conformal_mask].flatten()[group_mask] * scale).clamp(
            max=model.sigma_max
        )
        empirical_coverage[group] = float(
            (conformal_target[conformal_mask].flatten()[group_mask] <= group_radius).float().mean()
        )
    scale_tensor = sigma.new_ones(sigma.shape[1:3])
    for source in range(scale_tensor.shape[0]):
        for joint in range(scale_tensor.shape[1]):
            scale_tensor[source, joint] = calibration.scales[f"source_{source}:{_region(joint)}"]
    calibrated_sigma = (sigma * scale_tensor[None]).clamp(model.sigma_min, model.sigma_max)
    joint_uncertainty = calibrated_sigma.mean(1)
    abstention_thresholds = {
        "body": float(torch.quantile(joint_uncertainty[:, :25].mean(-1), 0.9)),
        "left_hand": float(torch.quantile(joint_uncertainty[:, 25:40].mean(-1), 0.9)),
        "right_hand": float(torch.quantile(joint_uncertainty[:, 40:55].mean(-1), 0.9)),
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = output_path / "weights.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}, weights_path
    )
    calibration.write(output_path / "group_scales.json")
    metrics: dict[str, object] = {
        "schema_version": "1.0",
        "initial_loss": history[0],
        "final_loss": history[-1],
        "epochs": epochs,
        "nominal_coverage": calibration.nominal_coverage,
        "empirical_conformal_coverage": empirical_coverage,
        "empirical_conformal_coverage_unclipped": empirical_coverage_unclipped,
        "model_fit_clips": train_ids,
        "conformal_clips": sorted(conformal_ids),
        "abstention_thresholds_m": abstention_thresholds,
        "seed": seed,
        "feature_dim": x.shape[-1],
        "hidden_dim": 32,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": sha256_file(model_path),
        "weights_sha256": sha256_file(weights_path),
        "cache_hashes": cache_hashes,
        "gt_file_count": len(gt_hashes),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(torch_device),
        "residual_alignment": "per_frame_source0_pelvis_translation",
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "history.jsonl").write_text(
        "".join(
            json.dumps({"epoch": index, "loss": loss}) + "\n" for index, loss in enumerate(history)
        ),
        encoding="utf-8",
    )
    return metrics
