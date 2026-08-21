from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..data.manifest import load_manifest
from ..evaluation.bootstrap import paired_hierarchical_bootstrap
from ..evaluation.dynamics import dynamics_errors
from ..evaluation.geometric import pa_mpvpe_mm, tr_v2v_mm
from ..evaluation.uncertainty import risk_coverage_curve, spearman_risk_error
from ..io.predictions import PredictionArtifact
from ..utils.hashing import sha256_file


def load_obj_vertices(path: str | Path) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    vertex_text = " ".join(line[2:] for line in text.splitlines() if line.startswith("v "))
    values = np.fromstring(vertex_text, sep=" ", dtype=np.float32)
    if not values.size or values.size % 3:
        raise ValueError(f"OBJ has no vertices: {path}")
    return torch.from_numpy(values.reshape(-1, 3))


def _load_regions(
    upper_path: str | Path, left_path: str | Path, right_path: str | Path
) -> dict[str, torch.Tensor]:
    return {
        "upper_body": torch.from_numpy(np.load(upper_path)).long(),
        "left_hand": torch.from_numpy(np.load(left_path)).long(),
        "right_hand": torch.from_numpy(np.load(right_path)).long(),
    }


def _load_joint_regressor(model_path: str | Path) -> torch.Tensor:
    """Load evaluator geometry without importing the fitting/model implementation."""
    try:
        import smplx
    except ImportError as exc:
        raise RuntimeError("SGNify evaluation requires the locally licensed smplx package") from exc
    path = Path(model_path)
    model = smplx.SMPLX(
        str(path),
        gender="neutral",
        ext=path.suffix.lstrip("."),
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    )
    return model.J_regressor.detach().cpu().float()


def cache_sgnify_ground_truth(manifest_path: str, gt_root: str, output_root: str) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    gt_root_path = Path(gt_root)
    output_path = Path(output_root)
    frames = 0
    for item in manifest:
        vertices = []
        hashes: dict[str, str] = {}
        for frame_id in item.frame_ids:
            path = gt_root_path / item.clip_id / f"{frame_id * 2:05d}.obj"
            if not path.is_file():
                raise FileNotFoundError(path)
            vertices.append(load_obj_vertices(path))
            hashes[str(path)] = sha256_file(path)
        clip_output = output_path / item.clip_id
        clip_output.mkdir(parents=True, exist_ok=True)
        tensor_path = clip_output / "vertices.safetensors"
        save_file(
            {
                "frame_ids": torch.tensor(item.frame_ids, dtype=torch.int64),
                "vertices": torch.stack(vertices),
            },
            tensor_path,
        )
        metadata = {
            "schema_version": "1.0",
            "clip_id": item.clip_id,
            "manifest_item_sha256": item.sha256,
            "artifact_sha256": sha256_file(tensor_path),
            "source_hashes": hashes,
        }
        (clip_output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frames += len(item.frame_ids)
    return {"clips": len(manifest), "frames": frames}


def _load_cached_ground_truth(root: Path, item: Any) -> torch.Tensor:
    clip_root = root / item.clip_id
    tensor_path = clip_root / "vertices.safetensors"
    metadata = json.loads((clip_root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["manifest_item_sha256"] != item.sha256:
        raise ValueError(f"GT cache manifest mismatch for {item.clip_id}")
    if metadata["artifact_sha256"] != sha256_file(tensor_path):
        raise ValueError(f"GT cache hash mismatch for {item.clip_id}")
    values = load_file(tensor_path)
    if values["frame_ids"].tolist() != item.frame_ids:
        raise ValueError(f"GT cache frame mismatch for {item.clip_id}")
    return values["vertices"]


def evaluate_sgnify(
    manifest_path: str,
    prediction_root: str,
    gt_root: str,
    model_path: str,
    upper_indices: str,
    left_indices: str,
    right_indices: str,
    output_root: str,
    gt_cache_root: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    prediction_root_path = Path(prediction_root)
    gt_root_path = Path(gt_root)
    gt_cache_root_path = Path(gt_cache_root) if gt_cache_root else None
    regions = _load_regions(upper_indices, left_indices, right_indices)
    model_hash = sha256_file(model_path)
    joint_regressor = _load_joint_regressor(model_path)
    frame_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for item in manifest:
        prediction, metadata = PredictionArtifact.load(prediction_root_path / item.clip_id)
        if prediction.frame_ids.tolist() != item.frame_ids:
            raise ValueError(f"completeness failure for {item.clip_id}")
        if prediction.vertices is None:
            raise ValueError(f"prediction has no vertices for {item.clip_id}")
        if prediction.vertices.shape[1] != joint_regressor.shape[1]:
            raise ValueError(f"SMPL-X vertex count mismatch for {item.clip_id}")
        if metadata.get("smplx_model_sha256") != model_hash:
            raise ValueError(f"prediction SMPL-X model hash mismatch for {item.clip_id}")
        if metadata.get("coordinate_convention") != "opencv_x_right_y_down_z_forward":
            raise ValueError(f"prediction coordinate convention mismatch for {item.clip_id}")
        if gt_cache_root_path is not None:
            target = _load_cached_ground_truth(gt_cache_root_path, item)
        else:
            ground_truth = []
            for frame_id in item.frame_ids:
                gt_path = gt_root_path / item.clip_id / f"{frame_id * 2:05d}.obj"
                if not gt_path.is_file():
                    raise FileNotFoundError(f"ground truth missing for manifest frame: {gt_path}")
                ground_truth.append(load_obj_vertices(gt_path))
            target = torch.stack(ground_truth)
        prediction_vertices = prediction.vertices.float()
        if target.shape != prediction_vertices.shape:
            raise ValueError(
                f"prediction/GT vertex shape mismatch for {item.clip_id}: "
                f"{prediction_vertices.shape} vs {target.shape}"
            )
        clip_metrics: dict[str, list[float]] = {}
        clip_scalar_metrics: dict[str, float] = {}
        for region_index, (region_name, indices) in enumerate(regions.items()):
            tr_values = tr_v2v_mm(prediction_vertices[:, indices], target[:, indices])
            pa_values = pa_mpvpe_mm(prediction_vertices[:, indices], target[:, indices])
            clip_metrics[f"tr_v2v_{region_name}_mm"] = tr_values.tolist()
            clip_metrics[f"pa_mpvpe_{region_name}_mm"] = pa_values.tolist()
            risk = prediction.risk_score[:, region_index].float()
            clip_metrics[f"abstain_{region_name}"] = (
                prediction.abstain[:, region_index].float().tolist()
            )
            curve = risk_coverage_curve(tr_values, risk)
            clip_scalar_metrics[f"aurc_{region_name}"] = float(curve["aurc"])
            clip_scalar_metrics[f"risk_error_spearman_{region_name}"] = spearman_risk_error(
                tr_values, risk
            )
            order = torch.argsort(risk)
            for fixed_coverage in (0.8, 0.9):
                keep = max(1, int(np.ceil(len(order) * fixed_coverage)))
                clip_scalar_metrics[f"selective_{int(fixed_coverage * 100)}_{region_name}_mm"] = (
                    float(tr_values[order[:keep]].mean())
                )
            clip_metrics[f"risk_{region_name}"] = risk.tolist()
        pred_joints = torch.einsum("jv,tvc->tjc", joint_regressor, prediction_vertices)
        target_joints = torch.einsum("jv,tvc->tjc", joint_regressor, target)
        dynamic = dynamics_errors(pred_joints, target_joints, item.fps)
        for name, values in dynamic.items():
            clip_metrics[name] = values.tolist()
        if prediction.contacts is not None and prediction.contact_probability is not None:
            clip_metrics["contact_active_fraction"] = prediction.contacts.float().mean(-1).tolist()
            clip_metrics["contact_probability_mean"] = (
                prediction.contact_probability.float().mean(-1).tolist()
            )
            collision_pairs = ((27, 45), (30, 48), (33, 51), (36, 54), (39, 42))
            distances = torch.stack(
                [
                    torch.linalg.vector_norm(pred_joints[:, left] - pred_joints[:, right], dim=-1)
                    for left, right in collision_pairs
                ],
                dim=-1,
            )
            clip_metrics["collision_proxy_penetration_mm"] = (
                torch.relu(0.008 - distances).max(-1).values.mul(1000).tolist()
            )
            transitions = prediction.contacts[1:] != prediction.contacts[:-1]
            clip_scalar_metrics["contact_switches_per_edge"] = float(
                transitions.float().sum(0).mean()
            )
        for frame_index, frame_id in enumerate(item.frame_ids):
            row: dict[str, Any] = {
                "clip_id": item.clip_id,
                "signer_id": item.signer_id,
                "frame_id": frame_id,
            }
            for name, values in clip_metrics.items():
                row[name] = values[frame_index] if frame_index < len(values) else None
            frame_rows.append(row)
        clip_row: dict[str, Any] = {
            "clip_id": item.clip_id,
            "signer_id": item.signer_id,
            "frames": len(item.frame_ids),
        }
        for name, values in clip_metrics.items():
            clip_row[name] = float(np.mean(values)) if values else None
        clip_row.update(clip_scalar_metrics)
        clip_rows.append(clip_row)

    aggregate: dict[str, Any] = {
        "clips": len(clip_rows),
        "frames": len(frame_rows),
        "coverage": len(frame_rows) / sum(len(item.frame_ids) for item in manifest),
    }
    numeric_names = [
        name
        for name in clip_rows[0]
        if name not in {"clip_id", "signer_id", "frames"}
        and all(row[name] is not None for row in clip_rows)
    ]
    for name in numeric_names:
        aggregate[name] = float(np.mean([row[name] for row in clip_rows]))
        frame_values = [
            float(row[name]) for row in frame_rows if name in row and row[name] is not None
        ]
        if frame_values:
            aggregate[f"micro_{name}"] = float(np.mean(frame_values))
    aggregate["primary_aggregation"] = "equal_weight_clip_macro"
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(frame_rows, output / "per_frame.csv")
    _write_csv(clip_rows, output / "per_clip.csv")
    (output / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    evaluation_metadata = {
        "schema_version": "1.0",
        "manifest_sha256": sha256_file(manifest_path),
        "smplx_model_sha256": model_hash,
        "region_sha256": {
            "upper_body": sha256_file(upper_indices),
            "left_hand": sha256_file(left_indices),
            "right_hand": sha256_file(right_indices),
        },
        "coverage_gate_passed": aggregate["coverage"] == 1.0,
        "alignment": {
            "primary": "translation-aligned vertex-to-vertex",
            "secondary": "per-frame Procrustes-aligned MPVPE",
        },
    }
    (output / "evaluation_metadata.json").write_text(
        json.dumps(evaluation_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def compare_clip_metrics(
    candidate_csv: str | Path,
    baseline_csv: str | Path,
    metric: str,
    output: str | Path,
    replicates: int = 10000,
) -> dict[str, float]:
    candidate_rows = _read_csv(candidate_csv)
    baseline_rows = _read_csv(baseline_csv)
    candidate = {row["clip_id"]: row for row in candidate_rows}
    baseline = {row["clip_id"]: row for row in baseline_rows}
    if set(candidate) != set(baseline):
        raise ValueError("paired comparison requires identical clip IDs")
    clip_ids = sorted(candidate)
    signer_ids = np.asarray([candidate[clip]["signer_id"] for clip in clip_ids])
    # Unknown signer IDs are replaced by clip IDs so clustering remains conservative and valid.
    signer_ids = np.asarray(
        [
            clip if signer == "unknown" else signer
            for clip, signer in zip(clip_ids, signer_ids, strict=True)
        ]
    )
    result = paired_hierarchical_bootstrap(
        np.asarray([float(candidate[clip][metric]) for clip in clip_ids]),
        np.asarray([float(baseline[clip][metric]) for clip in clip_ids]),
        signer_ids,
        np.asarray(clip_ids),
        replicates=replicates,
    )
    report = {
        "metric": metric,
        "candidate_minus_baseline": result.point_estimate,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "replicates": replicates,
    }
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
