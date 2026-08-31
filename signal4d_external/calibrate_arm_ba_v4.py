"""Calibrate arm-only 2D bundle adjustment on external How2Sign splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.refine_how2sign_targets import (
    _teacher_observations,
    create_smplx_model,
)
from phase2_refiner.provenance import sha256_file
from signal4d_external.arm_ba_v4_core import ARM_JOINTS, fit_arm_batch
from signal4d_external.calibrate_dposerx_body_v3 import (
    _errors,
    _load_split,
    _manifest_rows,
    _selection,
    _summary,
)
from signal4d_external.nlf_v2_core import geodesic_blend


BLENDS = (0.1, 0.25, 0.5, 0.75, 1.0)
COVERAGES = (0.1, 0.25, 0.5, 1.0)
DIRECTIONS = ("low", "high")


def _accepted_clips(rows: list[dict[str, Any]]) -> list[Any]:
    accepted = set(_load_split(rows)["clip_id"].tolist())
    return [load_cache_clip(row["cache_path"]) for row in rows if row["clip_id"] in accepted]


def _fit_split(
    model: Any,
    clips: list[Any],
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    outputs = []
    reports = []
    for start in range(0, len(clips), batch_size):
        batch = clips[start : start + batch_size]
        teacher = [_teacher_observations(clip) for clip in batch]
        candidate, batch_reports = fit_arm_batch(
            model,
            batch,
            observed=np.stack([item[0] for item in teacher]),
            confidence=np.stack([item[1] for item in teacher]),
            valid=np.stack([item[2] for item in teacher]),
            device=device,
            projection="how2sign",
            projection_aux=(
                np.stack([item[4] for item in teacher]),
                np.stack([item[5] for item in teacher]),
            ),
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            max_degrees=args.max_degrees,
        )
        outputs.append(candidate)
        reports.extend(batch_reports)
        print(f"[arm-ba-v4] {min(start + len(batch), len(clips))}/{len(clips)}", flush=True)
    return np.concatenate(outputs), reports


def _movement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = second[:, ARM_JOINTS] @ np.swapaxes(first[:, ARM_JOINTS], -1, -2)
    return np.rad2deg(
        np.linalg.norm(
            Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
        )
    ).reshape(len(first), len(ARM_JOINTS)).mean(axis=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    manifests = {
        "selection": args.selection_manifest.resolve(),
        "gate": args.gate_manifest.resolve(),
    }
    rows = {
        "selection": _manifest_rows(manifests["selection"], "val"),
        "gate": _manifest_rows(manifests["gate"], "test"),
    }
    clips = {name: _accepted_clips(value) for name, value in rows.items()}
    model = create_smplx_model(args.model_folder.resolve(), device)
    model.requires_grad_(False)
    candidate, fit_reports = {}, {}
    for split in ("selection", "gate"):
        candidate[split], fit_reports[split] = _fit_split(
            model, clips[split], device, args.batch_size, args
        )
    matrices, baseline_error, candidate_matrix = {}, {}, {}
    metadata = {}
    for split in ("selection", "gate"):
        initial = np.concatenate([clip.init_axis_angle for clip in clips[split]])
        target = np.concatenate([clip.target_axis_angle for clip in clips[split]])
        valid = np.concatenate([clip.target_rotation_valid for clip in clips[split]])
        signers = np.concatenate(
            [
                np.asarray([row["signer"]] * len(clip.frame_names))
                for row, clip in zip(
                    [r for r in rows[split] if r["clip_id"] in {x.clip_id for x in clips[split]}],
                    clips[split],
                    strict=True,
                )
            ]
        )
        base = Rotation.from_rotvec(initial.reshape(-1, 3)).as_matrix().reshape(-1, 51, 3, 3)
        target_matrix = Rotation.from_rotvec(target.reshape(-1, 3)).as_matrix().reshape(-1, 51, 3, 3)
        cand = Rotation.from_rotvec(candidate[split].reshape(-1, 3)).as_matrix().reshape(-1, 51, 3, 3)
        matrices[split] = {"base": base, "target": target_matrix}
        candidate_matrix[split] = cand
        baseline_error[split] = _errors(base[:, :21], target_matrix[:, :21], valid[:, :21])
        metadata[split] = {"valid": valid, "signers": signers}

    candidates = []
    for blend in BLENDS:
        blended = geodesic_blend(
            matrices["selection"]["base"], candidate_matrix["selection"], blend
        )
        error = _errors(
            blended[:, :21],
            matrices["selection"]["target"][:, :21],
            metadata["selection"]["valid"][:, :21],
        )
        movement = _movement(matrices["selection"]["base"], blended)
        for direction in DIRECTIONS:
            for coverage in COVERAGES:
                if coverage == 1.0:
                    threshold = float("inf") if direction == "low" else float("-inf")
                else:
                    quantile = coverage if direction == "low" else 1.0 - coverage
                    threshold = float(np.quantile(movement, quantile))
                chosen = _selection(movement, direction, threshold)
                candidates.append(
                    {
                        "blend": blend,
                        "direction": direction,
                        "coverage": coverage,
                        "movement_threshold_deg": threshold,
                        **_summary(
                            baseline_error["selection"],
                            error,
                            chosen,
                            metadata["selection"]["signers"],
                        ),
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
    gate_blended = geodesic_blend(
        matrices["gate"]["base"], candidate_matrix["gate"], float(selected["blend"])
    )
    gate_error = _errors(
        gate_blended[:, :21],
        matrices["gate"]["target"][:, :21],
        metadata["gate"]["valid"][:, :21],
    )
    gate_movement = _movement(matrices["gate"]["base"], gate_blended)
    gate_chosen = _selection(
        gate_movement,
        str(selected["direction"]),
        float(selected["movement_threshold_deg"]),
    )
    gate = _summary(
        baseline_error["gate"],
        gate_error,
        gate_chosen,
        metadata["gate"]["signers"],
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
        "schema_version": "signal4d.external_arm_ba_v4.v1",
        "decision": decision,
        "method": "arm-only observation-guided bundle adjustment",
        "sgnify_training_or_selection_reads": 0,
        "parameters": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "max_degrees": args.max_degrees,
            "arm_joints": ARM_JOINTS.tolist(),
            "observation_joints": [15, 50],
            "blends": list(BLENDS),
            "coverages": list(COVERAGES),
            "directions": list(DIRECTIONS),
        },
        "selected": selected,
        "gate": gate,
        "top_selection_candidates": sorted(candidates, key=lambda row: row["gain_deg"], reverse=True)[:20],
        "mean_reprojection_gain": {
            split: float(np.mean([item["relative_gain"] for item in fit_reports[split]]))
            for split in fit_reports
        },
        "external_split_counts": {
            split: {
                "clips": len(clips[split]),
                "frames": int(sum(len(clip.frame_names) for clip in clips[split])),
                "signers": sorted(set(metadata[split]["signers"].tolist())),
            }
            for split in clips
        },
        "model_folder": str(args.model_folder.resolve()),
        "model_sha256": sha256_file(args.model_folder.resolve() / "smplx" / "SMPLX_NEUTRAL.npz"),
        "manifests": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in manifests.items()
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
    parser.add_argument("--model-folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-degrees", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
