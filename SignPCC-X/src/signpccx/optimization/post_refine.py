from __future__ import annotations

from pathlib import Path
import csv
import io
import json

import numpy as np

from signpccx.data.manifest import read_jsonl
from signpccx.io import atomic_write_json, sha256_file
from signpccx.io import atomic_write_text
from signpccx.optimization.hypotheses import twist_wrist
from signpccx.geometry.contact_regions import nearest_region, write_contact_region_cache


_SIDES = {
    "left": {
        "body_slot": 19,
        "joint_ids": (20, 37, 25, 28, 34, 31),
        "anchor_ids": (6, 9, 10, 11, 12, 13),
    },
    "right": {
        "body_slot": 20,
        "joint_ids": (21, 52, 40, 43, 49, 46),
        "anchor_ids": (7, 20, 21, 22, 23, 24),
    },
}


def _project(points, intrinsics):
    projected = intrinsics @ points.transpose(1, 2)
    return projected[:, :2].transpose(1, 2) / projected[:, 2:3].transpose(1, 2).clamp_min(1e-6)


def _signed_area(points):
    wrist, index, pinky = points[:, 0], points[:, 2], points[:, 5]
    a, b = index - wrist, pinky - wrist
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _candidate_scores(
    predicted_joints,
    intrinsics,
    observed_uv,
    observed_xyz,
    confidence,
    degrees: float,
):
    import torch

    predicted_uv = _project(predicted_joints, intrinsics)
    weights = confidence.clamp_min(0.0)
    reprojection = (
        torch.linalg.vector_norm(predicted_uv - observed_uv, dim=-1) * weights
    ).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    predicted_centered = predicted_joints - predicted_joints[:, :1]
    observed_centered = observed_xyz - observed_xyz[:, :1]
    teacher = torch.linalg.vector_norm(predicted_centered - observed_centered, dim=-1).mean(dim=1)
    observed_sign = torch.sign(_signed_area(observed_uv))
    predicted_sign = torch.sign(_signed_area(predicted_uv))
    chirality = ((observed_sign != 0) & (predicted_sign != observed_sign)).to(predicted_joints.dtype)
    angle_prior = predicted_joints.new_full((len(predicted_joints),), abs(float(degrees)) / 30.0)
    # Each component is dimensionless. The conservative angle prior prevents a
    # weak 2-D cue from selecting a large twist against both 3-D teachers.
    total = reprojection / 20.0 + teacher / 0.03 + 0.25 * chirality + 0.05 * angle_prior
    return torch.stack((reprojection, teacher * 1000.0, chirality, total), dim=1), predicted_uv


def refine_palm_hypotheses(
    source_fit_root: Path,
    manifest_root: Path,
    h4wpp_frame_cache: Path,
    model_root: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    signs: set[str] | None = None,
    degrees: tuple[float, ...] = (-30.0, 0.0, 30.0),
) -> dict[str, object]:
    """Run deterministic K0 palm/chirality selection on parameter-complete fits.

    Ground truth and evaluator regions are deliberately absent. Candidate scores
    use only frozen H4W++ 2-D anchors, H4W++ 3-D anchors and a fixed twist prior.
    """
    import smplx
    import torch

    if 0.0 not in degrees:
        raise ValueError("Palm hypotheses must include the zero-twist baseline")
    model = smplx.create(
        str(model_root), model_type="smplx", gender="neutral", num_betas=10,
        use_pca=False, use_face_contour=True,
    ).to(device)
    model.eval()
    with torch.no_grad():
        neutral = model(return_verts=True).joints[0]
        axes = {
            side: neutral[spec["joint_ids"][3]] - neutral[spec["joint_ids"][0]]
            for side, spec in _SIDES.items()
        }
    boundary = torch.tensor((1.0, -1.0, -1.0), dtype=torch.float32, device=device)
    parameter_keys = (
        "betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
        "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
    )
    reports = []
    manifest_paths = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        unknown = signs - {path.stem for path in manifest_paths}
        if unknown:
            raise ValueError(f"Unknown signs: {sorted(unknown)}")
        manifest_paths = [path for path in manifest_paths if path.stem in signs]
    for manifest_path in manifest_paths:
        sign = manifest_path.stem
        records = read_jsonl(manifest_path)
        source = source_fit_root / "clips" / sign / "mesh_parametric_final.npz"
        destination = output_root / "clips" / sign / "mesh_parametric_final.npz"
        report_path = destination.with_suffix(".json")
        if destination.exists() or report_path.exists():
            if not destination.is_file() or not report_path.is_file():
                raise RuntimeError(f"Incomplete resumable palm output for {sign}")
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if cached.get("sha256") != sha256_file(destination):
                raise RuntimeError(f"Invalid resumable palm output for {sign}")
            reports.append(cached)
            continue
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
        if not all(key in arrays for key in parameter_keys):
            raise KeyError(f"{source}: parameter-complete state required")
        expected_ids = np.asarray([record.source_frame_id for record in records], dtype=np.int64)
        if not np.array_equal(arrays["frame_ids"], expected_ids):
            raise RuntimeError(f"{sign}: source fit frame IDs do not match manifest")
        caches = []
        for record in records:
            cache_path = h4wpp_frame_cache / "clips" / sign / f"{record.source_frame_id:06d}.npz"
            caches.append(np.load(cache_path, allow_pickle=False))
        intrinsics = torch.as_tensor(
            np.stack([item["K_crop"] for item in caches]), dtype=torch.float32, device=device
        )
        body_pose = torch.as_tensor(arrays["body_pose"], dtype=torch.float32, device=device).clone()
        fixed = {
            key: torch.as_tensor(arrays[key], dtype=torch.float32, device=device)
            for key in parameter_keys if key != "body_pose"
        }
        side_logs = {}
        with torch.no_grad():
            for side, spec in _SIDES.items():
                joint_ids = list(spec["joint_ids"])
                anchor_ids = list(spec["anchor_ids"])
                observed_uv = torch.as_tensor(
                    np.stack([item["anchor_uv_observed"][anchor_ids] for item in caches]),
                    dtype=torch.float32, device=device,
                )
                observed_xyz = torch.as_tensor(
                    np.stack([item["init_anchor_cam"][anchor_ids] for item in caches]),
                    dtype=torch.float32, device=device,
                )
                confidence = torch.as_tensor(
                    np.stack([
                        item["anchor_uv_confidence"][anchor_ids] * item["anchor_valid"][anchor_ids]
                        for item in caches
                    ]), dtype=torch.float32, device=device,
                )
                scores = []
                chirality_signs = []
                start = int(spec["body_slot"]) * 3
                base_wrist = body_pose[:, start:start + 3].clone()
                for value in degrees:
                    candidate_body = body_pose.clone()
                    candidate_body[:, start:start + 3] = twist_wrist(
                        base_wrist, axes[side].expand_as(base_wrist), value
                    )
                    output = model(body_pose=candidate_body, return_verts=True, **fixed)
                    candidate_joints = output.joints[:, joint_ids] * boundary
                    candidate_score, candidate_uv = _candidate_scores(
                        candidate_joints, intrinsics, observed_uv, observed_xyz, confidence, value
                    )
                    scores.append(candidate_score)
                    chirality_signs.append(torch.sign(_signed_area(candidate_uv)))
                score_tensor = torch.stack(scores, dim=1)
                selected = score_tensor[:, :, 3].argmin(dim=1)
                candidate_wrists = torch.stack([
                    twist_wrist(base_wrist, axes[side].expand_as(base_wrist), value)
                    for value in degrees
                ], dim=1)
                body_pose[:, start:start + 3] = candidate_wrists[
                    torch.arange(len(records), device=device), selected
                ]
                observed_sign = torch.sign(_signed_area(observed_uv))
                baseline_index = degrees.index(0.0)
                baseline_sign = chirality_signs[baseline_index]
                selected_sign = torch.stack(chirality_signs, dim=1)[
                    torch.arange(len(records), device=device), selected
                ]
                valid_sign = observed_sign != 0
                side_logs[side] = {
                    "scores": score_tensor.cpu().numpy().astype(np.float32),
                    "selected": selected.cpu().numpy().astype(np.int64),
                    "baseline_chirality_mismatches": int(((baseline_sign != observed_sign) & valid_sign).sum()),
                    "selected_chirality_mismatches": int(((selected_sign != observed_sign) & valid_sign).sum()),
                }
            final = model(body_pose=body_pose, return_verts=True, **fixed).vertices * boundary
        for item in caches:
            item.close()
        arrays["body_pose"] = body_pose.cpu().numpy().astype(np.float32)
        arrays["mesh_parametric"] = final.cpu().numpy().astype(np.float32)
        arrays["palm_hypothesis_degrees"] = np.asarray(degrees, dtype=np.float32)
        for side in _SIDES:
            arrays[f"{side}_palm_candidate_scores"] = side_logs[side]["scores"]
            arrays[f"{side}_palm_selected_index"] = side_logs[side]["selected"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        report = {
            "schema_version": "signpccx.palm-hypotheses.v1",
            "sign": sign,
            "frames": len(records),
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "degrees": list(degrees),
            "score_components": ["reprojection_px", "teacher_disagreement_mm", "chirality_mismatch", "total"],
            "objective_uses_ground_truth": False,
            "objective_uses_evaluator_upper_body_mask": False,
            "left_selection_counts": np.bincount(side_logs["left"]["selected"], minlength=len(degrees)).tolist(),
            "right_selection_counts": np.bincount(side_logs["right"]["selected"], minlength=len(degrees)).tolist(),
            "left_baseline_chirality_mismatches": side_logs["left"]["baseline_chirality_mismatches"],
            "left_selected_chirality_mismatches": side_logs["left"]["selected_chirality_mismatches"],
            "right_baseline_chirality_mismatches": side_logs["right"]["baseline_chirality_mismatches"],
            "right_selected_chirality_mismatches": side_logs["right"]["selected_chirality_mismatches"],
            "sha256": sha256_file(destination),
        }
        atomic_write_json(report_path, report)
        reports.append(report)
    summary = {
        "schema_version": "signpccx.palm-hypotheses-summary.v1",
        "source_fit_root": str(source_fit_root.resolve()),
        "h4wpp_frame_cache": str(h4wpp_frame_cache.resolve()),
        "degrees": list(degrees),
        "signs": len(reports),
        "frames": sum(int(item["frames"]) for item in reports),
        "selection_counts": {
            side: np.sum([item[f"{side}_selection_counts"] for item in reports], axis=0).astype(int).tolist()
            for side in _SIDES
        },
        "chirality_mismatches": {
            side: {
                "baseline": sum(int(item[f"{side}_baseline_chirality_mismatches"]) for item in reports),
                "selected": sum(int(item[f"{side}_selected_chirality_mismatches"]) for item in reports),
            }
            for side in _SIDES
        },
        "items": reports,
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow((
        "sign", "frame_id", "left_selected_degree", "right_selected_degree",
        "left_candidate_scores", "right_candidate_scores",
    ))
    for report in reports:
        path = output_root / "clips" / str(report["sign"]) / "mesh_parametric_final.npz"
        with np.load(path, allow_pickle=False) as archive:
            frame_ids = archive["frame_ids"]
            candidate_degrees = archive["palm_hypothesis_degrees"]
            left_selected = archive["left_palm_selected_index"]
            right_selected = archive["right_palm_selected_index"]
            left_scores = archive["left_palm_candidate_scores"]
            right_scores = archive["right_palm_candidate_scores"]
        for index, frame_id in enumerate(frame_ids):
            writer.writerow((
                report["sign"], int(frame_id), float(candidate_degrees[left_selected[index]]),
                float(candidate_degrees[right_selected[index]]),
                json.dumps(left_scores[index].tolist(), separators=(",", ":")),
                json.dumps(right_scores[index].tolist(), separators=(",", ":")),
            ))
    atomic_write_text(output_root / "candidate_scores.csv", stream.getvalue())
    atomic_write_json(output_root / "palm_hypotheses_summary.json", summary)
    return summary


def _per_frame_contact_distance(a, b):
    import torch

    distances = torch.cdist(a, b)
    return 0.5 * (
        distances.min(dim=2).values.mean(dim=1)
        + distances.min(dim=1).values.mean(dim=1)
    )


def refine_hand_contact(
    source_fit_root: Path,
    manifest_root: Path,
    h4wpp_frame_cache: Path,
    model_root: Path,
    mano_smplx_ids: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    signs: set[str] | None = None,
    confidence_threshold: float = 0.70,
    target_distance_m: float = 0.003,
    steps: int = 40,
    learning_rate: float = 0.001,
) -> dict[str, object]:
    """Confidence-gated hand--hand contact ablation on saved canonical states."""
    import pickle
    import smplx
    import torch

    model = smplx.create(
        str(model_root), model_type="smplx", gender="neutral", num_betas=10,
        use_pca=False, use_face_contour=True,
    ).to(device)
    model.eval()
    boundary = torch.tensor((1.0, -1.0, -1.0), dtype=torch.float32, device=device)
    with mano_smplx_ids.open("rb") as handle:
        mano_ids = pickle.load(handle)
    hand_ids = {
        side: np.asarray(mano_ids[f"{side}_hand"], dtype=np.int64)
        for side in ("left", "right")
    }
    tip_landmarks = {
        "left": (5361, 4933, 5058, 5169, 5286),
        "right": (8079, 7669, 7794, 7905, 8022),
    }
    with torch.no_grad():
        neutral = model(return_verts=True).vertices[0].cpu().numpy()
    regions = {
        f"{side}_{finger}_tip": nearest_region(
            neutral, neutral[tip_id], hand_ids[side], k=24
        )
        for side in ("left", "right")
        for finger, tip_id in zip(("thumb", "index", "middle", "ring", "pinky"), tip_landmarks[side])
    }
    region_cache = output_root.parent / "contact_regions.npz"
    if not region_cache.exists():
        write_contact_region_cache(
            region_cache, regions, model_root / "smplx" / "SMPLX_NEUTRAL.npz"
        )
    parameter_keys = (
        "betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
        "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
    )
    manifest_paths = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        unknown = signs - {path.stem for path in manifest_paths}
        if unknown:
            raise ValueError(f"Unknown signs: {sorted(unknown)}")
        manifest_paths = [path for path in manifest_paths if path.stem in signs]
    reports = []
    for manifest_path in manifest_paths:
        sign = manifest_path.stem
        records = read_jsonl(manifest_path)
        source = source_fit_root / "clips" / sign / "mesh_parametric_final.npz"
        destination = output_root / "clips" / sign / "mesh_parametric_final.npz"
        report_path = destination.with_suffix(".json")
        if destination.exists() or report_path.exists():
            if not destination.is_file() or not report_path.is_file():
                raise RuntimeError(f"Incomplete resumable contact output for {sign}")
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if cached.get("sha256") != sha256_file(destination):
                raise RuntimeError(f"Invalid resumable contact output for {sign}")
            reports.append(cached)
            continue
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
        if not all(key in arrays for key in parameter_keys):
            raise KeyError(f"{source}: parameter-complete state required")
        frame_ids = np.asarray([record.source_frame_id for record in records], dtype=np.int64)
        if not np.array_equal(arrays["frame_ids"], frame_ids):
            raise RuntimeError(f"{sign}: source fit frame IDs do not match manifest")
        proposals = []
        left_sparse, right_sparse = hand_ids["left"][::8], hand_ids["right"][::8]
        all_anchor_ids = [6, *range(8, 19), 7, *range(19, 30)]
        for frame_index, (mesh, frame_id) in enumerate(zip(arrays["mesh_parametric"], frame_ids)):
            cache_path = h4wpp_frame_cache / "clips" / sign / f"{frame_id:06d}.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                teacher_mesh = np.asarray(cache["mesh_hybrid_init"], dtype=np.float32)
                anchor_confidence = float(np.mean(
                    cache["anchor_uv_confidence"][all_anchor_ids]
                    * cache["anchor_valid"][all_anchor_ids]
                ))
            base_distance = float(np.linalg.norm(
                mesh[left_sparse, None] - mesh[right_sparse][None], axis=2
            ).min())
            teacher_distance = float(np.linalg.norm(
                teacher_mesh[left_sparse, None] - teacher_mesh[right_sparse][None], axis=2
            ).min())
            geometry = max(base_distance, teacher_distance)
            proximity = float(np.clip((0.015 - geometry) / 0.010, 0.0, 1.0))
            agreement = float(np.exp(-abs(base_distance - teacher_distance) / 0.005))
            confidence = 0.50 * proximity + 0.25 * agreement + 0.25 * anchor_confidence
            if confidence < confidence_threshold:
                continue
            best = None
            for side, opposing in (("left", "right"), ("right", "left")):
                opposing_ids = hand_ids[opposing][::4]
                for region_name, region_ids in regions.items():
                    if not region_name.startswith(f"{side}_"):
                        continue
                    distances = np.linalg.norm(
                        mesh[region_ids, None] - mesh[opposing_ids][None], axis=2
                    )
                    flat = int(distances.argmin())
                    distance = float(distances.reshape(-1)[flat])
                    opposing_vertex = int(opposing_ids[flat % len(opposing_ids)])
                    opposing_region = hand_ids[opposing][np.argsort(
                        np.linalg.norm(mesh[hand_ids[opposing]] - mesh[opposing_vertex], axis=1),
                        kind="stable",
                    )[:48]]
                    candidate = (distance, side, region_name, region_ids, opposing_region)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            assert best is not None
            proposals.append({
                "frame_index": frame_index,
                "frame_id": int(frame_id),
                "confidence": confidence,
                "base_min_distance_m": base_distance,
                "teacher_min_distance_m": teacher_distance,
                "direction": f"{best[1]}_tip_to_opposing_hand",
                "tip_region": best[2],
                "region_a": np.asarray(best[3], dtype=np.int64),
                "region_b": np.asarray(best[4], dtype=np.int64),
            })
        initial_distances = []
        final_distances = []
        if proposals:
            active = np.asarray([item["frame_index"] for item in proposals], dtype=np.int64)
            base_vertices = torch.as_tensor(
                arrays["mesh_parametric"][active], dtype=torch.float32, device=device
            )
            base_body = torch.as_tensor(arrays["body_pose"][active], dtype=torch.float32, device=device)
            base_left = torch.as_tensor(arrays["left_hand_pose"][active], dtype=torch.float32, device=device)
            base_right = torch.as_tensor(arrays["right_hand_pose"][active], dtype=torch.float32, device=device)
            body_delta = torch.nn.Parameter(torch.zeros_like(base_body))
            left_delta = torch.nn.Parameter(torch.zeros_like(base_left))
            right_delta = torch.nn.Parameter(torch.zeros_like(base_right))
            body_mask = torch.zeros_like(base_body)
            body_mask[:, 19 * 3:21 * 3] = 1.0
            fixed = {
                key: torch.as_tensor(arrays[key][active], dtype=torch.float32, device=device)
                for key in parameter_keys if key not in ("body_pose", "left_hand_pose", "right_hand_pose")
            }
            region_a = torch.as_tensor(
                np.stack([item["region_a"] for item in proposals]), dtype=torch.long, device=device
            )
            region_b = torch.as_tensor(
                np.stack([item["region_b"] for item in proposals]), dtype=torch.long, device=device
            )
            confidence = torch.as_tensor(
                [item["confidence"] for item in proposals], dtype=torch.float32, device=device
            )
            left_anchor_ids = torch.as_tensor(left_sparse, dtype=torch.long, device=device)
            right_anchor_ids = torch.as_tensor(right_sparse, dtype=torch.long, device=device)

            def forward_vertices():
                return model(
                    body_pose=base_body + body_delta * body_mask,
                    left_hand_pose=base_left + left_delta,
                    right_hand_pose=base_right + right_delta,
                    return_verts=True,
                    **fixed,
                ).vertices * boundary

            def gather(vertices, ids):
                return torch.gather(vertices, 1, ids.unsqueeze(2).expand(-1, -1, 3))

            optimizer = torch.optim.Adam((body_delta, left_delta, right_delta), lr=learning_rate)
            best_loss = float("inf")
            best_state = None
            with torch.no_grad():
                first_vertices = forward_vertices()
                initial = _per_frame_contact_distance(
                    gather(first_vertices, region_a), gather(first_vertices, region_b)
                )
                initial_distances = initial.cpu().numpy().tolist()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                vertices = forward_vertices()
                distances = _per_frame_contact_distance(
                    gather(vertices, region_a), gather(vertices, region_b)
                )
                contact = (
                    confidence * torch.square(distances - target_distance_m)
                ).sum() / confidence.sum().clamp_min(1e-6)
                absolute_anchor = 0.5 * (
                    torch.square(vertices[:, left_anchor_ids] - base_vertices[:, left_anchor_ids]).mean()
                    + torch.square(vertices[:, right_anchor_ids] - base_vertices[:, right_anchor_ids]).mean()
                )
                penetration = torch.square(torch.relu(0.001 - distances)).mean()
                pose_anchor = (
                    torch.square(body_delta * body_mask).mean()
                    + torch.square(left_delta).mean()
                    + torch.square(right_delta).mean()
                )
                loss = contact + 2.0 * absolute_anchor + 0.20 * penetration + 1e-4 * pose_anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{sign}: non-finite contact loss")
                loss.backward()
                optimizer.step()
                current = float(loss.detach())
                if current < best_loss:
                    best_loss = current
                    best_state = (
                        body_delta.detach().clone(), left_delta.detach().clone(), right_delta.detach().clone()
                    )
            if best_state is None:
                raise RuntimeError(f"{sign}: contact refinement produced no valid state")
            with torch.no_grad():
                body_delta.copy_(best_state[0])
                left_delta.copy_(best_state[1])
                right_delta.copy_(best_state[2])
                final_vertices = forward_vertices()
                final = _per_frame_contact_distance(
                    gather(final_vertices, region_a), gather(final_vertices, region_b)
                )
                final_distances = final.cpu().numpy().tolist()
            arrays["body_pose"][active] = (
                base_body + body_delta * body_mask
            ).detach().cpu().numpy().astype(np.float32)
            arrays["left_hand_pose"][active] = (
                base_left + left_delta
            ).detach().cpu().numpy().astype(np.float32)
            arrays["right_hand_pose"][active] = (
                base_right + right_delta
            ).detach().cpu().numpy().astype(np.float32)
            arrays["mesh_parametric"][active] = final_vertices.cpu().numpy().astype(np.float32)
        arrays["contact_active"] = np.zeros(len(records), dtype=bool)
        arrays["contact_confidence"] = np.zeros(len(records), dtype=np.float32)
        arrays["contact_initial_distance_m"] = np.full(len(records), np.nan, dtype=np.float32)
        arrays["contact_final_distance_m"] = np.full(len(records), np.nan, dtype=np.float32)
        for item, initial, final in zip(proposals, initial_distances, final_distances):
            index = int(item["frame_index"])
            arrays["contact_active"][index] = True
            arrays["contact_confidence"][index] = item["confidence"]
            arrays["contact_initial_distance_m"][index] = initial
            arrays["contact_final_distance_m"][index] = final
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        serializable_proposals = [{
            key: value for key, value in item.items() if key not in ("region_a", "region_b")
        } for item in proposals]
        for item, initial, final in zip(serializable_proposals, initial_distances, final_distances):
            item["optimized_initial_distance_m"] = initial
            item["optimized_final_distance_m"] = final
            item["target_distance_m"] = target_distance_m
        report = {
            "schema_version": "signpccx.contact-refinement.v1",
            "sign": sign,
            "frames": len(records),
            "active_frames": len(proposals),
            "confidence_threshold": confidence_threshold,
            "target_distance_m": target_distance_m,
            "steps": steps,
            "learning_rate": learning_rate,
            "objective_uses_ground_truth": False,
            "objective_uses_evaluator_upper_body_mask": False,
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "proposals": serializable_proposals,
            "sha256": sha256_file(destination),
        }
        atomic_write_json(report_path, report)
        reports.append(report)
    active = sum(int(item["active_frames"]) for item in reports)
    initial_values = [
        proposal["optimized_initial_distance_m"]
        for item in reports for proposal in item["proposals"]
    ]
    final_values = [
        proposal["optimized_final_distance_m"]
        for item in reports for proposal in item["proposals"]
    ]
    summary = {
        "schema_version": "signpccx.contact-refinement-summary.v1",
        "source_fit_root": str(source_fit_root.resolve()),
        "confidence_threshold": confidence_threshold,
        "target_distance_m": target_distance_m,
        "signs": len(reports),
        "frames": sum(int(item["frames"]) for item in reports),
        "active_frames": active,
        "mean_absolute_target_error_initial_mm": None if not active else float(
            np.mean(np.abs(np.asarray(initial_values) - target_distance_m)) * 1000.0
        ),
        "mean_absolute_target_error_final_mm": None if not active else float(
            np.mean(np.abs(np.asarray(final_values) - target_distance_m)) * 1000.0
        ),
        "items": reports,
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow((
        "sign", "frame_id", "confidence", "direction", "tip_region",
        "initial_distance_mm", "final_distance_mm", "target_distance_mm",
    ))
    for item in reports:
        for proposal in item["proposals"]:
            writer.writerow((
                item["sign"], proposal["frame_id"], proposal["confidence"],
                proposal["direction"], proposal["tip_region"],
                1000.0 * proposal["optimized_initial_distance_m"],
                1000.0 * proposal["optimized_final_distance_m"],
                1000.0 * proposal["target_distance_m"],
            ))
    atomic_write_text(output_root / "contact_frames.csv", stream.getvalue())
    atomic_write_json(output_root / "contact_refinement_summary.json", summary)
    return summary
