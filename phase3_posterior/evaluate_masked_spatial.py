"""Formal decoded SMPL-X masked-spatial evaluation for Phase 3 Stage R3."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from phase2_refiner.data.corruptions import refresh_rotation_features
from phase2_refiner.geometry.smplx_decode import decode_smplx_sequence
from phase2_refiner.render import create_smplx_model
from phase3_posterior.config import load_config
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.gates import stage_decision
from phase3_posterior.geometry.state_adapter import matrices_to_state, state_to_matrices
from phase3_posterior.losses.diffusion import SubVPSDE
from phase3_posterior.masked_spatial import (
    FORMAL_MASKS,
    REGION_JOINTS,
    fixed_condition_mask,
    inject_fixed_rotation_corruption,
    to_device,
)
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.sample import sample_candidates
from phase3_posterior.training import load_weights, seed_everything


def _empty() -> dict[str, float]:
    return {"sum": 0.0, "count": 0.0, "frames": 0.0}


def _accumulate(total: dict[str, float], error: torch.Tensor, frames: torch.Tensor) -> None:
    selected = error[frames]
    if selected.numel() == 0:
        return
    total["sum"] += float(selected.sum())
    total["count"] += int(selected.numel())
    total["frames"] += int(frames.sum())


def _mean(total: dict[str, float]) -> float | None:
    return total["sum"] / total["count"] if total["count"] else None


def _face(batch: dict) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].float()
        for key in ("jaw_pose", "leye_pose", "reye_pose", "expression")
    }


def _decode(model, matrix: torch.Tensor, batch: dict) -> torch.Tensor:
    vertices, _ = decode_smplx_sequence(
        model,
        matrix.float(),
        batch["betas"].float(),
        batch["global_orient"].float(),
        batch["transl"].float(),
        **_face(batch),
    )
    return vertices


def _vertex_error(
    value: torch.Tensor, target: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    return (
        torch.linalg.vector_norm(
            value.index_select(-2, ids) - target.index_select(-2, ids), dim=-1
        )
        * 1000.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("SMPLer-X/common/utils/human_model_files"),
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("data/evaluation_from_author/data/data"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("batch-size and steps must be positive")
    seed_everything(args.seed)
    config = load_config(args.config)
    device = torch.device(args.device)
    dataset = Phase3Dataset(
        config["data"]["val_index"],
        int(config["model"]["max_frames"]),
        training=False,
        seed=int(config.get("seed", 42)) + 1,
        input_dim=int(config["model"].get("observation_dim", 45)),
        identity_target=bool(config["data"].get("identity_target", False)),
    )
    full_clip_count = len(dataset)
    if args.max_clips is not None:
        if args.max_clips < 1:
            raise ValueError("max-clips must be positive")
        dataset = Subset(dataset, range(min(args.max_clips, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, int(config["training"].get("workers", 0))),
        collate_fn=collate_phase3,
    )
    model = RelationalDiffusionPosterior(config["model"]).to(device).eval()
    checkpoint = load_weights(model, str(args.checkpoint.resolve()), strict=True)
    body_model = create_smplx_model(args.model_folder.resolve(), device)
    body_model.requires_grad_(False)
    with (args.assets_root / "MANO_SMPLX_vertex_ids.pkl").open("rb") as handle:
        hand_ids = pickle.load(handle, encoding="latin1")
    vertex_ids = {
        "ubody": torch.as_tensor(
            np.load(
                args.assets_root
                / "sgnify_part_segm_above_pelvis_joint"
                / "upper_body_minus_face.npy"
            ),
            device=device,
            dtype=torch.long,
        ),
        "lhand": torch.as_tensor(
            hand_ids["left_hand"], device=device, dtype=torch.long
        ),
        "rhand": torch.as_tensor(
            hand_ids["right_hand"], device=device, dtype=torch.long
        ),
    }
    sde = SubVPSDE(
        **{key: config["diffusion"][key] for key in ("beta_min", "beta_max", "eps")}
    )
    clean = {
        region: {"baseline": _empty(), "prediction": _empty()}
        for region in REGION_JOINTS
    }
    masked = {
        mask.name: {"baseline": _empty(), "prediction": _empty()}
        for mask in FORMAL_MASKS
    }
    observed_exact = True
    clips = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = to_device(raw_batch, device)
        clips += int(batch["frame_valid"].shape[0])
        observation_valid = batch["frame_valid"][..., None].expand(-1, -1, 51)
        target_vertices = _decode(body_model, batch["target_matrix"], batch)
        initial_vertices = _decode(body_model, batch["initial_matrix"], batch)
        clean_state = sample_candidates(
            model,
            batch,
            sde,
            candidates=2,
            steps=args.steps,
            seed=args.seed + batch_index * 1009,
            condition_mask=observation_valid,
        )[:, 1]
        observed_exact &= torch.equal(
            clean_state[observation_valid], batch["initial_state"][observation_valid]
        )
        clean_vertices = _decode(body_model, state_to_matrices(clean_state), batch)
        for region, joints in REGION_JOINTS.items():
            frame_valid = batch["frame_valid"] & batch["target_rotation_valid"][
                ..., list(joints)
            ].all(-1)
            _accumulate(
                clean[region]["baseline"],
                _vertex_error(initial_vertices, target_vertices, vertex_ids[region]),
                frame_valid,
            )
            _accumulate(
                clean[region]["prediction"],
                _vertex_error(clean_vertices, target_vertices, vertex_ids[region]),
                frame_valid,
            )
        for mask_index, mask in enumerate(FORMAL_MASKS):
            condition = fixed_condition_mask(observation_valid, mask)
            corrupted_matrix, corruption_mask = inject_fixed_rotation_corruption(
                batch["initial_matrix"].float(),
                batch["target_rotation_valid"],
                batch["frame_valid"],
                mask,
                seed=args.seed + batch_index * 1009 + mask_index * 100_003,
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
            prediction_state = sample_candidates(
                model,
                corrupted_batch,
                sde,
                candidates=2,
                steps=args.steps,
                seed=args.seed + batch_index * 1009 + mask_index * 100_003,
                condition_mask=condition,
                rotation_hint_mask=corruption_mask if use_hints else None,
            )[:, 1]
            prediction_vertices = _decode(
                body_model, state_to_matrices(prediction_state), batch
            )
            joints = REGION_JOINTS[mask.region]
            frame_valid = batch["frame_valid"] & batch["target_rotation_valid"][
                ..., list(joints)
            ].all(-1)
            frame_valid &= corruption_mask.any(-1)
            _accumulate(
                masked[mask.name]["baseline"],
                _vertex_error(
                    _decode(body_model, corrupted_matrix, batch),
                    target_vertices,
                    vertex_ids[mask.region],
                ),
                frame_valid,
            )
            _accumulate(
                masked[mask.name]["prediction"],
                _vertex_error(
                    prediction_vertices, target_vertices, vertex_ids[mask.region]
                ),
                frame_valid,
            )
        if (batch_index + 1) % 10 == 0 or clips == len(dataset):
            print(
                json.dumps(
                    {
                        "evaluation_progress": {
                            "batches": batch_index + 1,
                            "clips": clips,
                            "expected_clips": len(dataset),
                        }
                    }
                ),
                flush=True,
            )

    clean_metrics = {}
    clean_regressions = []
    for region, values in clean.items():
        before = _mean(values["baseline"])
        after = _mean(values["prediction"])
        if before is None or after is None:
            regression = None
        elif before <= 1e-3:
            regression = 0.0 if after <= 1e-3 else 1.0e30
        else:
            regression = after / before - 1.0
        clean_metrics[region] = {
            "baseline_mm": before,
            "prediction_mm": after,
            "regression": regression,
            "frames": int(values["baseline"]["frames"]),
        }
        if regression is not None:
            clean_regressions.append(regression)
    by_mask = {}
    regional_recoveries: dict[str, list[float]] = {
        region: [] for region in REGION_JOINTS
    }
    for mask in FORMAL_MASKS:
        values = masked[mask.name]
        before = _mean(values["baseline"])
        after = _mean(values["prediction"])
        recovery = (
            None
            if before in (None, 0.0) or after is None
            else 1.0 - after / before
        )
        by_mask[mask.name] = {
            "region": mask.region,
            "baseline_mm": before,
            "prediction_mm": after,
            "recovery": recovery,
            "frames": int(values["baseline"]["frames"]),
        }
        if recovery is not None:
            regional_recoveries[mask.region].append(recovery)
    recovery = {
        region: min(values) if values else -1.0e30
        for region, values in regional_recoveries.items()
    }
    max_clean_regression = (
        max(clean_regressions)
        if len(clean_regressions) == len(REGION_JOINTS)
        else 1.0e30
    )
    formal_full_coverage = args.max_clips is None and clips == full_clip_count
    metrics = {
        "schema_version": 1,
        "stage": "R3 masked spatial diffusion",
        "units": "millimetres",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "weights": "EMA",
        "clips": clips,
        "expected_clips": full_clip_count,
        "formal_full_coverage": formal_full_coverage,
        "sampling_steps": args.steps,
        "seed": args.seed,
        "clean_observed_state_exact": observed_exact,
        "clean": clean_metrics,
        "by_mask": by_mask,
        "recovery": recovery,
        "max_clean_regression": max_clean_regression,
        "provenance": {
            "config_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "validation_manifest_sha256": sha256_file(
                config["data"]["val_index"]
            ),
        },
    }
    decision = stage_decision("g3", metrics)
    if not formal_full_coverage:
        decision["passed"] = False
        decision["checks"]["formal_full_coverage"] = {
            "passed": False,
            "actual": clips,
            "requirement": str(full_clip_count),
        }
    else:
        decision["checks"]["formal_full_coverage"] = {
            "passed": True,
            "actual": clips,
            "requirement": str(full_clip_count),
        }
    metrics["decision"] = decision
    atomic_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
