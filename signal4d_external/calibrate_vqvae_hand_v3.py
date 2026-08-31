"""Externally calibrate sign-VQ-VAE hand manifold projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from phase2_refiner.provenance import sha256_file
from signal4d_external.calibrate_dposerx_body_v3 import (
    _load_split,
    _manifest_rows,
    _selection,
    _summary,
)
from signal4d_external.nlf_v2_core import geodesic_blend


BLENDS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
COVERAGES = (0.05, 0.1, 0.25, 0.5, 1.0)
DIRECTIONS = ("low", "high")
HAND_SLICES = {"left": slice(21, 36), "right": slice(36, 51)}


def _hand_errors(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    result = np.full(len(prediction), np.nan, dtype=np.float64)
    for frame in range(len(result)):
        selected = np.flatnonzero(valid[frame])
        if len(selected) < 8:
            continue
        delta = prediction[frame, selected] @ np.swapaxes(
            target[frame, selected], -1, -2
        )
        result[frame] = float(
            np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
                )
            ).mean()
        )
    return result


def _movement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = second @ np.swapaxes(first, -1, -2)
    return np.rad2deg(
        np.linalg.norm(
            Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
        )
    ).reshape(len(first), 15).mean(axis=1)


@torch.no_grad()
def _reconstruct(model: Any, poses: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    result = []
    for start in range(0, len(poses), batch_size):
        batch = torch.from_numpy(poses[start : start + batch_size].reshape(-1, 45)).to(
            device
        )
        repeated = batch[:, None, :].expand(-1, 8, -1).contiguous()
        reconstruction, _, _ = model(repeated)
        result.append(reconstruction[:, reconstruction.shape[1] // 2].cpu().numpy())
    return np.concatenate(result).reshape(-1, 15, 3)


def _hand_arrays(values: dict[str, np.ndarray], hand: str) -> dict[str, np.ndarray]:
    region = HAND_SLICES[hand]
    pose = values["pose_all"][:, region]
    target = values["target_all"][:, region]
    valid = values["valid_all"][:, region]
    return {"pose": pose, "target": target, "valid": valid}


def _load_full_split(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    # Reuse the body calibrator's strict provenance and official-split filter,
    # then reload only the already accepted cache paths to retain all 51 joints.
    body = _load_split(rows)
    accepted = set(body["clip_id"].tolist())
    pose_parts, target_parts, valid_parts = [], [], []
    from phase2_refiner.data.cache_schema import load_cache_clip

    for row in rows:
        if row["clip_id"] not in accepted:
            continue
        cache = load_cache_clip(row["cache_path"])
        pose_parts.append(np.asarray(cache.init_axis_angle[:, :51], dtype=np.float32))
        target_parts.append(np.asarray(cache.target_axis_angle[:, :51], dtype=np.float32))
        valid_parts.append(np.asarray(cache.target_rotation_valid[:, :51], dtype=bool))
    return {
        **body,
        "pose_all": np.concatenate(pose_parts),
        "target_all": np.concatenate(target_parts),
        "valid_all": np.concatenate(valid_parts),
    }


def _select_policy(
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
    movement: np.ndarray,
    signers: np.ndarray,
    blend: float,
) -> list[dict[str, Any]]:
    rows = []
    for direction in DIRECTIONS:
        for coverage in COVERAGES:
            if coverage == 1.0:
                threshold = float("inf") if direction == "low" else float("-inf")
            else:
                quantile = coverage if direction == "low" else 1.0 - coverage
                threshold = float(np.quantile(movement, quantile))
            chosen = _selection(movement, direction, threshold)
            rows.append(
                {
                    "blend": blend,
                    "direction": direction,
                    "coverage": coverage,
                    "movement_threshold_deg": threshold,
                    **_summary(baseline_error, candidate_error, chosen, signers),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    from signhposer_vqvae.loaders import load_signhposer_vqvae

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model, config = load_signhposer_vqvae(
        ckpt_path=str(args.checkpoint.resolve()), map_location=str(device)
    )
    model = model.to(device).eval()
    rows = {
        "selection": _manifest_rows(args.selection_manifest.resolve(), "val"),
        "gate": _manifest_rows(args.gate_manifest.resolve(), "test"),
    }
    data = {name: _load_full_split(value) for name, value in rows.items()}
    results = {}
    for hand in ("left", "right"):
        arrays = {name: _hand_arrays(values, hand) for name, values in data.items()}
        matrices = {}
        baseline_error = {}
        projected = {}
        for split, values in arrays.items():
            base = Rotation.from_rotvec(values["pose"].reshape(-1, 3)).as_matrix() \
                .reshape(-1, 15, 3, 3)
            target = Rotation.from_rotvec(values["target"].reshape(-1, 3)).as_matrix() \
                .reshape(-1, 15, 3, 3)
            matrices[split] = {"base": base, "target": target}
            baseline_error[split] = _hand_errors(base, target, values["valid"])
            reconstructed = _reconstruct(model, values["pose"], args.batch_size, device)
            projected[split] = Rotation.from_rotvec(reconstructed.reshape(-1, 3)) \
                .as_matrix().reshape(-1, 15, 3, 3)

        candidates = []
        for blend in BLENDS:
            candidate = geodesic_blend(
                matrices["selection"]["base"], projected["selection"], blend
            )
            error = _hand_errors(
                candidate,
                matrices["selection"]["target"],
                arrays["selection"]["valid"],
            )
            movement = _movement(matrices["selection"]["base"], candidate)
            candidates.extend(
                _select_policy(
                    baseline_error["selection"],
                    error,
                    movement,
                    data["selection"]["signer"],
                    blend,
                )
            )
        eligible = [
            row
            for row in candidates
            if row["gain_deg"] > 0.0
            and row["selection_fraction"] >= 0.045
            and row["median_delta_deg"] <= 0.0
            and row["p95_regression_deg"] <= 0.5
            and row["worst_signer_gain_deg"] >= -0.1
        ]
        selected = max(
            eligible or candidates,
            key=lambda row: (row["gain_deg"], -row["p95_regression_deg"], -row["blend"]),
        )
        gate_candidate = geodesic_blend(
            matrices["gate"]["base"], projected["gate"], float(selected["blend"])
        )
        gate_error = _hand_errors(
            gate_candidate, matrices["gate"]["target"], arrays["gate"]["valid"]
        )
        gate_movement = _movement(matrices["gate"]["base"], gate_candidate)
        gate_selected = _selection(
            gate_movement,
            str(selected["direction"]),
            float(selected["movement_threshold_deg"]),
        )
        gate = _summary(
            baseline_error["gate"],
            gate_error,
            gate_selected,
            data["gate"]["signer"],
        )
        decision = (
            "PASS"
            if bool(eligible)
            and gate["gain_deg"] > 0.0
            and gate["selection_fraction"] >= 0.025
            and gate["median_delta_deg"] <= 0.0
            and gate["p95_regression_deg"] <= 0.5
            and gate["worst_signer_gain_deg"] >= -0.1
            else "FAIL"
        )
        results[hand] = {
            "decision": decision,
            "selected": selected,
            "gate": gate,
            "top_selection_candidates": sorted(
                candidates, key=lambda row: row["gain_deg"], reverse=True
            )[:20],
        }

    report = {
        "schema_version": "signal4d.external_vqvae_hand_v3.v1",
        "decision": "PASS" if any(x["decision"] == "PASS" for x in results.values()) else "FAIL",
        "method": "sign-VQ-VAE reconstruction with SO(3) blend and observable movement abstention",
        "training_data": "external How2Sign SMPLer-X hand poses only",
        "sgnify_training_or_selection_reads": 0,
        "seed": args.seed,
        "grid": {
            "blends": list(BLENDS),
            "coverages": list(COVERAGES),
            "directions": list(DIRECTIONS),
        },
        "regions": results,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_config": config,
        "manifests": {
            "selection": {
                "path": str(args.selection_manifest.resolve()),
                "sha256": sha256_file(args.selection_manifest),
            },
            "gate": {
                "path": str(args.gate_manifest.resolve()),
                "sha256": sha256_file(args.gate_manifest),
            },
        },
        "external_split_counts": {
            name: {
                "clips": int(len(set(values["clip_id"]))),
                "frames": int(len(values["pose"])),
                "signers": sorted(set(values["signer"].tolist())),
            }
            for name, values in data.items()
        },
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--gate-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
