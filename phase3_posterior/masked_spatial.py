"""Deterministic masked-spatial validation shared by R3 training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from phase2_refiner.data.corruptions import refresh_rotation_features
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from phase3_posterior.geometry.state_adapter import matrices_to_state, state_to_matrices
from phase3_posterior.losses.diffusion import SubVPSDE
from phase3_posterior.sample import sample_candidates


REGION_JOINTS = {
    "ubody": tuple(range(0, 21)),
    "lhand": tuple(range(21, 36)),
    "rhand": tuple(range(36, 51)),
}


@dataclass(frozen=True)
class SpatialMask:
    name: str
    region: str
    hidden_joints: tuple[int, ...]


FORMAL_MASKS = (
    SpatialMask("upper_body", "ubody", tuple(range(2, 21))),
    SpatialMask("left_hand", "lhand", tuple(range(21, 36))),
    SpatialMask("right_hand", "rhand", tuple(range(36, 51))),
    SpatialMask("left_finger_chain", "lhand", (21, 22, 23)),
    SpatialMask("right_finger_chain", "rhand", (36, 37, 38)),
    SpatialMask("left_wrist_attachment", "lhand", (19, 21)),
    SpatialMask("right_wrist_attachment", "rhand", (20, 36)),
)

SELECTION_MASKS = FORMAL_MASKS[:3]


def fixed_condition_mask(valid: torch.Tensor, mask: SpatialMask) -> torch.Tensor:
    if valid.ndim != 3 or valid.shape[-1] != 51:
        raise ValueError("valid must have shape (B,T,51)")
    result = valid.clone()
    result[..., list(mask.hidden_joints)] = False
    return result


def inject_fixed_rotation_corruption(
    initial_matrix: torch.Tensor,
    target_valid: torch.Tensor,
    frame_valid: torch.Tensor,
    mask: SpatialMask,
    *,
    seed: int,
    max_degrees: float = 35.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject a replayable local-rotation error into a declared hidden mask."""
    corruption_mask = torch.zeros_like(target_valid)
    corruption_mask[..., list(mask.hidden_joints)] = (
        target_valid[..., list(mask.hidden_joints)] & frame_valid[..., None]
    )
    return inject_masked_rotation_corruption(
        initial_matrix,
        corruption_mask,
        seed=seed,
        max_degrees=max_degrees,
    )


def inject_masked_rotation_corruption(
    initial_matrix: torch.Tensor,
    corruption_mask: torch.Tensor,
    *,
    seed: int,
    max_degrees: float = 35.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject replayable local rotations exactly where ``corruption_mask`` is true."""
    if max_degrees <= 0:
        raise ValueError("max_degrees must be positive")
    if corruption_mask.shape != initial_matrix.shape[:-2]:
        raise ValueError("corruption_mask must have shape (B,T,51)")
    output = initial_matrix.clone()
    generator = torch.Generator(device=output.device).manual_seed(seed)
    direction = torch.randn(
        output.shape[:-2] + (3,),
        device=output.device,
        dtype=output.dtype,
        generator=generator,
    )
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    magnitude = torch.rand(
        output.shape[:-2] + (1,),
        device=output.device,
        dtype=output.dtype,
        generator=generator,
    )
    residual = direction * magnitude * math.radians(max_degrees)
    corrupted = axis_angle_to_matrix(residual) @ output
    output = torch.where(
        corruption_mask[..., None, None], corrupted, output
    )
    return output, corruption_mask


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _empty_totals(masks: Iterable[SpatialMask]) -> dict[str, dict[str, float]]:
    return {
        mask.name: {"baseline_sum": 0.0, "prediction_sum": 0.0, "count": 0.0}
        for mask in masks
    }


@torch.no_grad()
def evaluate_rotation_proxy(
    model: torch.nn.Module,
    loader,
    sde: SubVPSDE,
    device: torch.device,
    *,
    steps: int,
    seed: int,
    max_batches: int,
    masks: tuple[SpatialMask, ...] = SELECTION_MASKS,
) -> dict:
    """Evaluate deterministic conditional samples in SO(3) for checkpoint ranking."""
    if steps < 1 or max_batches < 1:
        raise ValueError("steps and max_batches must be positive")
    totals = _empty_totals(masks)
    clean_totals = {
        region: {"baseline_sum": 0.0, "prediction_sum": 0.0, "count": 0.0}
        for region in REGION_JOINTS
    }
    was_training = model.training
    model.eval()
    clean_exact = True
    batches = 0
    try:
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = to_device(raw_batch, device)
            observation_valid = batch["frame_valid"][..., None].expand(-1, -1, 51)
            clean = sample_candidates(
                model,
                batch,
                sde,
                candidates=2,
                steps=steps,
                seed=seed + batch_index * 1009,
                condition_mask=observation_valid,
            )[:, 1]
            clean_exact &= torch.equal(
                clean[observation_valid], batch["initial_state"][observation_valid]
            )
            target_matrix = batch["target_matrix"].float()
            initial_matrix = batch["initial_matrix"].float()
            clean_matrix = state_to_matrices(clean)
            for region_name, region_joints in REGION_JOINTS.items():
                region = list(region_joints)
                valid = (
                    batch["target_rotation_valid"][..., region]
                    & batch["frame_valid"][..., None]
                )
                clean_before = geodesic_distance(
                    initial_matrix[..., region, :, :],
                    target_matrix[..., region, :, :],
                )[valid]
                clean_after = geodesic_distance(
                    clean_matrix[..., region, :, :],
                    target_matrix[..., region, :, :],
                )[valid]
                if clean_before.numel():
                    clean_totals[region_name]["baseline_sum"] += float(
                        clean_before.sum()
                    )
                    clean_totals[region_name]["prediction_sum"] += float(
                        clean_after.sum()
                    )
                    clean_totals[region_name]["count"] += int(clean_before.numel())
            for mask_index, mask in enumerate(masks):
                condition = fixed_condition_mask(observation_valid, mask)
                corrupted_matrix, corruption_mask = inject_fixed_rotation_corruption(
                    initial_matrix,
                    batch["target_rotation_valid"],
                    batch["frame_valid"],
                    mask,
                    seed=seed + batch_index * 1009 + mask_index * 100_003,
                )
                corrupted_batch = dict(batch)
                corrupted_batch["initial_matrix"] = corrupted_matrix
                corrupted_batch["initial_state"] = matrices_to_state(corrupted_matrix)
                use_hints = (
                    getattr(model.residual, "corruption_observation", None) is not None
                )
                if use_hints:
                    corrupted_batch["features"] = refresh_rotation_features(
                        batch["features"], corrupted_matrix
                    )
                condition |= (
                    batch["frame_valid"][..., None]
                    & ~batch["target_rotation_valid"]
                    & ~corruption_mask
                )
                prediction = sample_candidates(
                    model,
                    corrupted_batch,
                    sde,
                    candidates=2,
                    steps=steps,
                    seed=seed + batch_index * 1009 + mask_index * 100_003,
                    condition_mask=condition,
                    rotation_hint_mask=corruption_mask if use_hints else None,
                )[:, 1]
                prediction_matrix = state_to_matrices(prediction)
                region = list(REGION_JOINTS[mask.region])
                valid = (
                    batch["target_rotation_valid"][..., region]
                    & batch["frame_valid"][..., None]
                )
                before = geodesic_distance(
                    corrupted_matrix[..., region, :, :],
                    target_matrix[..., region, :, :],
                )[valid]
                after = geodesic_distance(
                    prediction_matrix[..., region, :, :],
                    target_matrix[..., region, :, :],
                )[valid]
                if before.numel():
                    totals[mask.name]["baseline_sum"] += float(before.sum())
                    totals[mask.name]["prediction_sum"] += float(after.sum())
                    totals[mask.name]["count"] += int(before.numel())
            batches += 1
    finally:
        model.train(was_training)

    by_mask = {}
    regional: dict[str, list[float]] = {name: [] for name in REGION_JOINTS}
    for mask in masks:
        values = totals[mask.name]
        count = int(values["count"])
        baseline = values["baseline_sum"] / count if count else None
        prediction = values["prediction_sum"] / count if count else None
        recovery = (
            None
            if baseline in (None, 0.0) or prediction is None
            else 1.0 - prediction / baseline
        )
        by_mask[mask.name] = {
            "region": mask.region,
            "baseline_radians": baseline,
            "prediction_radians": prediction,
            "recovery": recovery,
            "joint_frames": count,
        }
        if recovery is not None:
            regional[mask.region].append(recovery)
    recovery = {
        region: min(values) if values else None for region, values in regional.items()
    }
    clean_regression = {}
    for region, values in clean_totals.items():
        count = int(values["count"])
        baseline = values["baseline_sum"] / count if count else None
        prediction = values["prediction_sum"] / count if count else None
        if baseline is None or prediction is None:
            clean_regression[region] = None
        elif baseline <= 1e-6:
            clean_regression[region] = (
                0.0 if prediction <= 1e-6 else float("inf")
            )
        else:
            clean_regression[region] = prediction / baseline - 1.0
    available_clean = [
        value for value in clean_regression.values() if value is not None
    ]
    max_clean_regression = (
        max(available_clean)
        if len(available_clean) == len(REGION_JOINTS)
        else float("inf")
    )
    ratios = [1.0 - value for value in recovery.values() if value is not None]
    if len(ratios) != len(REGION_JOINTS):
        selection_score = float("inf")
    else:
        hard_gain = 1.0 - sum(ratios) / len(ratios)
        selection_score = (
            sum(ratios) / len(ratios)
            + 0.5
            * sum(
                max(0.0, value - 0.01)
                for value in clean_regression.values()
                if value is not None
            )
            + 0.25 * (1.0 - min(max(hard_gain, 0.0), 1.0))
        )
    return {
        "metric": "deterministic_masked_so3_checkpoint_selection",
        "batches": batches,
        "sampling_steps": steps,
        "clean_observed_state_exact": clean_exact,
        "clean_regression": clean_regression,
        "by_mask": by_mask,
        "recovery": recovery,
        "max_clean_regression": max_clean_regression,
        "selection_score": selection_score,
    }
