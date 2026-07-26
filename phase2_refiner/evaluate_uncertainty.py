"""Export matched U0/U1 residuals for the Phase 2 calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase2_refiner.config import load_config
from phase2_refiner.data.corruptions import apply_burst_corruption
from phase2_refiner.data.dataset import (
    U0_RELIABILITY,
    SequenceCacheDataset,
    collate_sequences,
)
from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.infer import _load_model
from phase2_refiner.provenance import sha256_file


REGION_GROUP = np.asarray(["body"] * 21 + ["left_hand"] * 15 + ["right_hand"] * 15)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def export_residuals(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    u1_config = load_config(args.u1_config)
    u0_config = load_config(args.u0_config)
    u1 = _load_model(u1_config, args.u1_checkpoint, device)
    u0 = _load_model(u0_config, args.u0_checkpoint, device)
    if not u1.predict_uncertainty:
        raise ValueError("U1 checkpoint does not have a learned uncertainty head")
    if u0.predict_uncertainty:
        raise ValueError("U0 checkpoint unexpectedly has a learned uncertainty head")
    if u0.max_frames != u1.max_frames:
        raise ValueError("U0 and U1 max_frames differ")

    real_audit = None
    exact_locked_a1 = False
    if args.real_residual_audit:
        with args.real_residual_audit.open("r", encoding="utf-8") as handle:
            real_audit = json.load(handle)
        if not real_audit.get("passed", False):
            raise ValueError("Real-residual audit did not pass")
        calibration_audit = real_audit.get("calibration")
        if calibration_audit is None:
            raise ValueError(
                "Real-residual audit does not include the calibration manifest"
            )
        if Path(calibration_audit["manifest"]).resolve() != args.manifest.resolve():
            raise ValueError(
                "Calibration manifest differs from the one in the real-residual audit"
            )
        if not real_audit.get("split_disjoint_verified", False):
            raise ValueError("Calibration source groups are not split-disjoint")
        exact_locked_a1 = all(
            bool((real_audit.get(split) or {}).get("locked_initializer_required"))
            for split in ("train", "validation", "calibration")
        )
        if not exact_locked_a1 and not args.diagnostic_only:
            raise ValueError(
                "Residual audit is a proxy audit, not an exact locked-A1 audit; "
                "use --diagnostic-only and do not claim G5"
            )
    elif not args.diagnostic_only:
        raise ValueError(
            "A passing --real-residual-audit is required; use --diagnostic-only "
            "only for synthetic engineering checks"
        )

    dataset = SequenceCacheDataset(
        str(args.manifest.resolve()),
        max_frames=u1.max_frames,
        training=False,
        input_dim=u1.token_embedding.input_projection.in_features,
        reprojection_residual_scale=float(
            u1_config.get("data", {}).get("reprojection_residual_scale", 10.0)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    corruption = u1_config.get("validation_corruption") or u1_config.get("corruption", {})
    corruption = dict(corruption)
    corruption["probability"] = 1.0
    torch.manual_seed(args.seed)
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "error",
            "log_variance",
            "u0_log_variance",
            "u1_corrupt_error",
            "u0_corrupt_error",
            "u1_clean_error",
            "u0_clean_error",
            "group",
        )
    }
    for batch in loader:
        batch = _to_device(batch, device)
        clean_features = batch["features"]
        clean_initial = batch["initial_matrix"]
        corrupt_features, corrupt_initial, _ = apply_burst_corruption(
            clean_features,
            clean_initial,
            batch["frame_valid"],
            target_rotation_valid=batch["target_rotation_valid"],
            **corruption,
        )
        u1_corrupt = u1(
            corrupt_features,
            corrupt_initial,
            batch["frame_valid"],
            batch["refine_mask"],
            batch["initial_joint_position"],
        )
        u0_corrupt = u0(
            corrupt_features,
            corrupt_initial,
            batch["frame_valid"],
            batch["refine_mask"],
            batch["initial_joint_position"],
        )
        u1_clean = u1(
            clean_features,
            clean_initial,
            batch["frame_valid"],
            batch["refine_mask"],
            batch["initial_joint_position"],
        )
        u0_clean = u0(
            clean_features,
            clean_initial,
            batch["frame_valid"],
            batch["refine_mask"],
            batch["initial_joint_position"],
        )
        mask = (
            batch["frame_valid"][:, :, None]
            & batch["target_rotation_valid"]
            & batch["refine_mask"][:, None, :]
        )
        errors = {
            "u1_corrupt_error": geodesic_distance(
                u1_corrupt["matrix"], batch["target_matrix"]
            ),
            "u0_corrupt_error": geodesic_distance(
                u0_corrupt["matrix"], batch["target_matrix"]
            ),
            "u1_clean_error": geodesic_distance(
                u1_clean["matrix"], batch["target_matrix"]
            ),
            "u0_clean_error": geodesic_distance(
                u0_clean["matrix"], batch["target_matrix"]
            ),
        }
        for key, value in errors.items():
            arrays[key].append(value[mask].cpu().numpy())
        arrays["error"].append(errors["u1_corrupt_error"][mask].cpu().numpy())
        arrays["log_variance"].append(
            u1_corrupt["log_variance"].squeeze(-1)[mask].cpu().numpy()
        )
        # U0 detector confidence is interpreted as precision and receives its
        # own scalar calibration in calibrate.py for a fair NLL comparison.
        u0_log_variance = -torch.log(
            corrupt_features[..., U0_RELIABILITY].clamp_min(1e-4)
        )
        arrays["u0_log_variance"].append(u0_log_variance[mask].cpu().numpy())
        group_grid = np.broadcast_to(
            REGION_GROUP[None, None, :], tuple(mask.shape)
        )
        arrays["group"].append(group_grid[mask.cpu().numpy()])

    payload = {key: np.concatenate(value) for key, value in arrays.items()}
    payload.update(
        {
            "source_kind": np.asarray(
                "real_residual_exact_a1"
                if exact_locked_a1
                else (
                    "proxy_residual_diagnostic"
                    if real_audit is not None
                    else "synthetic_diagnostic"
                )
            ),
            "split_disjoint_verified": np.asarray(
                real_audit is not None
                and bool(real_audit.get("split_disjoint_verified", False))
            ),
            "u1_checkpoint_sha256": np.asarray(sha256_file(args.u1_checkpoint)),
            "u0_checkpoint_sha256": np.asarray(sha256_file(args.u0_checkpoint)),
            "manifest_sha256": np.asarray(sha256_file(args.manifest)),
            "real_residual_audit_sha256": np.asarray(
                sha256_file(args.real_residual_audit)
                if args.real_residual_audit
                else ""
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    return {
        "output": str(args.output.resolve()),
        "observations": int(len(payload["error"])),
        "source_kind": str(payload["source_kind"]),
        "split_disjoint_verified": bool(payload["split_disjoint_verified"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--u1-config", type=Path, required=True)
    parser.add_argument("--u1-checkpoint", type=Path, required=True)
    parser.add_argument("--u0-config", type=Path, required=True)
    parser.add_argument("--u0-checkpoint", type=Path, required=True)
    parser.add_argument("--real-residual-audit", type=Path)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    print(json.dumps(export_residuals(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
