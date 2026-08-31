"""Externally calibrate a conservative sign-DPoser-X body projection.

The selection split is How2Sign ``val`` and the final gate is How2Sign
``test``.  No SGNify artifact is accepted or read by this program.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file
from signal4d_external.nlf_v2_core import ARMS, UPPER_BODY, geodesic_blend


TIMESTEPS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.125)
BLENDS = (0.05, 0.1, 0.25, 0.5, 1.0)
COVERAGES = (0.1, 0.25, 0.5, 1.0)
DIRECTIONS = ("low", "high")
FORBIDDEN_PARTS = {"sgnify", "evaluation_from_author", "smplx_gt"}


def _manifest_rows(path: Path, expected_split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("clips")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest has no clips: {path}")
    normalized = []
    for row in rows:
        cache_path = Path(row.get("cache_path", row.get("clip_path", ""))).resolve()
        cache_sha = row.get("cache_sha256", row.get("clip_sha256"))
        if not cache_path.is_file() or not cache_sha:
            raise ValueError(f"Invalid cache record: {row}")
        if sha256_file(cache_path) != cache_sha:
            raise ValueError(f"Cache hash mismatch: {cache_path}")
        if {part.lower() for part in cache_path.parts} & FORBIDDEN_PARTS:
            raise ValueError(f"Forbidden cache path: {cache_path}")
        normalized.append(
            {
                "clip_id": str(row["clip_id"]),
                "signer": str(row["signer"]),
                "source_group": str(row["source_group"]),
                "cache_path": cache_path,
                "cache_sha256": cache_sha,
                "frame_ids": row.get("frame_ids"),
                "expected_split": expected_split,
            }
        )
    return normalized


def _load_split(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    pose_parts = []
    target_parts = []
    valid_parts = []
    signer_parts = []
    group_parts = []
    clip_parts = []
    for row in rows:
        cache = load_cache_clip(row["cache_path"])
        metadata = json.loads(cache.metadata_json)
        if metadata.get("dataset") != "How2Sign":
            raise ValueError(f"Non-How2Sign cache: {row['cache_path']}")
        if int(metadata.get("sgnify_training_reads", 0)) != 0:
            raise ValueError(f"Cache reports SGNify reads: {row['cache_path']}")
        # The signer-disjoint source manifest spans official train/val clips.
        # Filter by immutable dataset metadata before reading any target array,
        # so the score-prior checkpoint is calibrated only on unseen val data.
        if metadata.get("official_split") != row["expected_split"]:
            continue
        if cache.target_axis_angle is None or cache.target_rotation_valid is None:
            raise ValueError(f"External target absent: {row['cache_path']}")
        # ``frame_ids`` are source-video identifiers, not array offsets.  The
        # cache itself is already materialized in exactly that manifest order.
        source_frame_ids = row["frame_ids"]
        if source_frame_ids is not None and len(source_frame_ids) != len(cache.frame_names):
            raise ValueError(f"Frame-id coverage mismatch: {row['cache_path']}")
        indices = np.arange(len(cache.frame_names), dtype=np.int64)
        count = len(indices)
        pose_parts.append(np.asarray(cache.init_axis_angle[indices, :21], dtype=np.float32))
        target_parts.append(
            np.asarray(cache.target_axis_angle[indices, :21], dtype=np.float32)
        )
        valid_parts.append(
            np.asarray(cache.target_rotation_valid[indices, :21], dtype=bool)
        )
        signer_parts.extend([row["signer"]] * count)
        group_parts.extend([row["source_group"]] * count)
        clip_parts.extend([row["clip_id"]] * count)
    return {
        "pose": np.concatenate(pose_parts),
        "target": np.concatenate(target_parts),
        "valid": np.concatenate(valid_parts),
        "signer": np.asarray(signer_parts),
        "source_group": np.asarray(group_parts),
        "clip_id": np.asarray(clip_parts),
    }


def _errors(
    prediction_matrix: np.ndarray,
    target_matrix: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.full(len(prediction_matrix), np.nan, dtype=np.float64)
    for frame in range(len(result)):
        selected = UPPER_BODY[valid[frame, UPPER_BODY]]
        if len(selected) < 4:
            continue
        delta = prediction_matrix[frame, selected] @ np.swapaxes(
            target_matrix[frame, selected], -1, -2
        )
        error = np.rad2deg(
            np.linalg.norm(
                Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
            )
        )
        weights = np.where(np.isin(selected, ARMS), 2.0, 1.0)
        result[frame] = float(np.average(error, weights=weights))
    return result


def _displacement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = second @ np.swapaxes(first, -1, -2)
    degrees = np.rad2deg(
        np.linalg.norm(
            Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
        )
    ).reshape(len(first), 21)
    return np.average(
        degrees[:, UPPER_BODY],
        axis=1,
        weights=np.where(np.isin(UPPER_BODY, ARMS), 2.0, 1.0),
    )


@torch.no_grad()
def _project(
    prior: Any,
    poses: np.ndarray,
    timestep: float,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(poses), batch_size):
        clean = torch.from_numpy(poses[start : start + batch_size].reshape(-1, 63)).to(
            prior._device
        )
        normalized = prior.Normalizer.offline_normalize(clean)
        t = torch.full(
            (len(normalized),), timestep, device=normalized.device, dtype=normalized.dtype
        )
        alpha, sigma_sq = prior.sde.return_alpha_sigma(t)
        noisy = alpha * normalized
        score = prior.score_fn(noisy, t, condition=None, mask=None)
        projected = (noisy + sigma_sq.reshape(-1, 1) * score) / alpha
        outputs.append(
            prior.Normalizer.offline_denormalize(projected)
            .reshape(-1, 21, 3)
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs)


def _summary(
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
    selected: np.ndarray,
    signers: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(baseline_error) & np.isfinite(candidate_error)
    chosen = finite & selected
    hybrid = baseline_error.copy()
    hybrid[chosen] = candidate_error[chosen]
    delta = hybrid[finite] - baseline_error[finite]
    signer_gain = {}
    for signer in sorted(set(signers[finite])):
        mask = finite & (signers == signer)
        signer_gain[str(signer)] = float(
            np.mean(baseline_error[mask]) - np.mean(hybrid[mask])
        )
    return {
        "frames": int(finite.sum()),
        "selected_frames": int(chosen.sum()),
        "selection_fraction": float(chosen.sum() / max(finite.sum(), 1)),
        "baseline_error_deg": float(np.mean(baseline_error[finite])),
        "hybrid_error_deg": float(np.mean(hybrid[finite])),
        "gain_deg": float(-np.mean(delta)),
        "median_delta_deg": float(np.median(delta)),
        "p95_regression_deg": float(np.quantile(delta, 0.95)),
        "improved_fraction": float(np.mean(delta < 0.0)),
        "signer_gain_deg": signer_gain,
        "worst_signer_gain_deg": float(min(signer_gain.values())),
    }


def _selection(displacement: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "low":
        return displacement <= threshold
    return displacement >= threshold


def run(args: argparse.Namespace) -> dict[str, Any]:
    # The legacy SMPLify-X tree uses top-level imports from its own directory.
    from signbposer_dposerx.loaders import load_signbposer_dposerx

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    prior = load_signbposer_dposerx(
        str(args.config.resolve()),
        str(args.checkpoint.resolve()),
        str(args.normalizer.resolve()),
        device=str(device),
        timestep_strategy="fixed",
    )
    split_rows = {
        "selection": _manifest_rows(args.selection_manifest.resolve(), "val"),
        "gate": _manifest_rows(args.gate_manifest.resolve(), "test"),
    }
    data = {name: _load_split(rows) for name, rows in split_rows.items()}
    matrices = {}
    baseline_errors = {}
    for name, values in data.items():
        matrices[name] = {
            "baseline": Rotation.from_rotvec(values["pose"].reshape(-1, 3))
            .as_matrix()
            .reshape(-1, 21, 3, 3),
            "target": Rotation.from_rotvec(values["target"].reshape(-1, 3))
            .as_matrix()
            .reshape(-1, 21, 3, 3),
        }
        baseline_errors[name] = _errors(
            matrices[name]["baseline"], matrices[name]["target"], values["valid"]
        )

    candidates = []
    cached = defaultdict(dict)
    for timestep in TIMESTEPS:
        for split in ("selection", "gate"):
            projected = _project(prior, data[split]["pose"], timestep, args.batch_size)
            cached[timestep][split] = Rotation.from_rotvec(projected.reshape(-1, 3)) \
                .as_matrix().reshape(-1, 21, 3, 3)
        for blend in BLENDS:
            base = matrices["selection"]["baseline"]
            candidate = geodesic_blend(base, cached[timestep]["selection"], blend)
            error = _errors(candidate, matrices["selection"]["target"], data["selection"]["valid"])
            movement = _displacement(base, candidate)
            for direction in DIRECTIONS:
                for coverage in COVERAGES:
                    if coverage == 1.0:
                        threshold = float("inf") if direction == "low" else float("-inf")
                    else:
                        quantile = coverage if direction == "low" else 1.0 - coverage
                        threshold = float(np.quantile(movement, quantile))
                    chosen = _selection(movement, direction, threshold)
                    summary = _summary(
                        baseline_errors["selection"], error, chosen, data["selection"]["signer"]
                    )
                    candidates.append(
                        {
                            "timestep": timestep,
                            "blend": blend,
                            "direction": direction,
                            "coverage": coverage,
                            "movement_threshold_deg": threshold,
                            **summary,
                        }
                    )
    eligible = [
        row
        for row in candidates
        if row["gain_deg"] > 0.0
        and row["selection_fraction"] >= 0.09
        and row["median_delta_deg"] <= 0.0
        and row["p95_regression_deg"] <= 0.5
        and row["worst_signer_gain_deg"] >= -0.1
    ]
    selected = max(
        eligible or candidates,
        key=lambda row: (row["gain_deg"], -row["p95_regression_deg"], -row["blend"]),
    )
    timestep = float(selected["timestep"])
    blend = float(selected["blend"])
    gate_base = matrices["gate"]["baseline"]
    gate_candidate = geodesic_blend(gate_base, cached[timestep]["gate"], blend)
    gate_error = _errors(
        gate_candidate, matrices["gate"]["target"], data["gate"]["valid"]
    )
    gate_movement = _displacement(gate_base, gate_candidate)
    gate_chosen = _selection(
        gate_movement,
        str(selected["direction"]),
        float(selected["movement_threshold_deg"]),
    )
    gate = _summary(
        baseline_errors["gate"], gate_error, gate_chosen, data["gate"]["signer"]
    )
    decision = (
        "PASS"
        if bool(eligible)
        and gate["gain_deg"] > 0.0
        and gate["selection_fraction"] >= 0.05
        and gate["median_delta_deg"] <= 0.0
        and gate["p95_regression_deg"] <= 0.5
        and gate["worst_signer_gain_deg"] >= -0.1
        else "FAIL"
    )
    report = {
        "schema_version": "signal4d.external_dposerx_body_v3.v1",
        "decision": decision,
        "method": "deterministic zero-noise Tweedie projection with observable movement abstention",
        "training_data": "How2Sign train only",
        "sgnify_training_or_selection_reads": 0,
        "seed": args.seed,
        "grid": {
            "timesteps": list(TIMESTEPS),
            "blends": list(BLENDS),
            "coverages": list(COVERAGES),
            "directions": list(DIRECTIONS),
        },
        "selected": selected,
        "gate": gate,
        "top_selection_candidates": sorted(
            candidates, key=lambda row: row["gain_deg"], reverse=True
        )[:20],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "normalizer": str(args.normalizer.resolve()),
        "normalizer_sha256": sha256_file(args.normalizer / "axis_normalize1.pt"),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
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
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
