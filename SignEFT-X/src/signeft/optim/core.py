from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from signeft.data.manifest import ManifestRecord, read_manifest
from signeft.gating.evidence_gate import FamilyDelta, accept_ubody
from signeft.gating.rollback import exact_rollback
from signeft.io.obj import load_obj, write_obj
from signeft.io_utils import atomic_write_json, sha256_file, tree_sha256
from signeft.model.kinematics import translation_aligned_hand_distance
from signeft.model.smplx_adapter import (
    BaselineBatch, TrustRegionSMPLX, load_mano_vertex_ids,
    resolve_joint_indices, semantic_vertex_ids,
)
from signeft.observations.common import atomic_savez


HEATMAP_CORE_NAMES = (
    "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)
NLF_EDGES_BY_NAME = (
    ("pelvis", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
)
U1_ACTIVE = ("spine1", "spine2", "spine3", "neck", "left_collar", "right_collar")
U2_ACTIVE = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
ALL_ACTIVE = U1_ACTIVE + U2_ACTIVE


@dataclass
class PoseObservationBatch:
    heatmaps: torch.Tensor
    valid: torch.Tensor
    weight: torch.Tensor
    crop_to_full: torch.Tensor
    covariance_full: torch.Tensor
    names: tuple[str, ...]


@dataclass
class NLFObservationBatch:
    joints: torch.Tensor
    valid: torch.Tensor
    uncertainty: torch.Tensor
    names: tuple[str, ...]


def _load_pose(records: Sequence[ManifestRecord], root: Path, device: str) -> PoseObservationBatch:
    arrays = []
    expected_names: tuple[str, ...] | None = None
    for record in records:
        path = root / record.sign_id / f"{record.source_frame_id:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(archive["joint_names"].tolist())
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise RuntimeError(f"pose joint schema drift: {path}")
            if str(archive["rgb_sha256"]) != record.sha256_rgb:
                raise RuntimeError(f"pose/RGB hash mismatch: {path}")
            scale = np.asarray(archive["heatmap_scale"], dtype=np.float32)[:, None, None]
            zero = np.asarray(archive["heatmap_zero"], dtype=np.float32)[:, None, None]
            heatmap = np.asarray(archive["heatmap_q"], dtype=np.float32) * scale + zero
            arrays.append({
                "heatmap": heatmap,
                "valid": np.asarray(archive["valid"], dtype=bool),
                "score": np.asarray(archive["score"], dtype=np.float32),
                "entropy": np.asarray(archive["entropy"], dtype=np.float32),
                "crop": np.asarray(archive["crop_to_full"], dtype=np.float32),
                "cov": np.asarray(archive["cov2d"], dtype=np.float32),
            })
    assert expected_names is not None
    mapping = resolve_joint_indices(expected_names, HEATMAP_CORE_NAMES)
    selected = [mapping[name] for name in HEATMAP_CORE_NAMES]
    heatmaps = torch.as_tensor(np.stack([item["heatmap"][selected] for item in arrays]), device=device)
    mass = heatmaps.sum(dim=(-2, -1), keepdim=True)
    valid = torch.as_tensor(np.stack([item["valid"][selected] for item in arrays]), device=device)
    valid &= mass.squeeze(-1).squeeze(-1) > 0
    heatmaps = heatmaps.clamp_min(0) / mass.clamp_min(1e-12)
    score = torch.as_tensor(np.stack([item["score"][selected] for item in arrays]), device=device)
    entropy = torch.as_tensor(np.stack([item["entropy"][selected] for item in arrays]), device=device)
    max_entropy = float(np.log(heatmaps.shape[-2] * heatmaps.shape[-1]))
    weight = score * (1.0 - entropy / max_entropy).clamp(0, 1) * valid
    return PoseObservationBatch(
        heatmaps=heatmaps,
        valid=valid,
        weight=weight,
        crop_to_full=torch.as_tensor(np.stack([item["crop"] for item in arrays]), device=device),
        covariance_full=torch.as_tensor(np.stack([item["cov"][selected] for item in arrays]), device=device),
        names=HEATMAP_CORE_NAMES,
    )


def _load_nlf(records: Sequence[ManifestRecord], root: Path, device: str) -> NLFObservationBatch:
    arrays = []
    expected_names: tuple[str, ...] | None = None
    for record in records:
        path = root / record.sign_id / f"{record.source_frame_id:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(archive["joint_names"].tolist())
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise RuntimeError(f"NLF joint schema drift: {path}")
            if str(archive["rgb_sha256"]) != record.sha256_rgb:
                raise RuntimeError(f"NLF/RGB hash mismatch: {path}")
            if str(archive["coord_frame"]) != "evaluator_camera_centered" or str(archive["unit"]) != "meter":
                raise RuntimeError(f"NLF coordinate contract mismatch: {path}")
            arrays.append({
                "joints": np.asarray(archive["joints3d"], dtype=np.float32),
                "valid": np.asarray(archive["valid"], dtype=bool),
                "uncertainty": np.asarray(archive["joint_uncertainty"], dtype=np.float32),
            })
    assert expected_names is not None
    return NLFObservationBatch(
        joints=torch.as_tensor(np.stack([item["joints"] for item in arrays]), device=device),
        valid=torch.as_tensor(np.stack([item["valid"] for item in arrays]), device=device),
        uncertainty=torch.as_tensor(np.stack([item["uncertainty"] for item in arrays]), device=device),
        names=expected_names,
    )


def project(joints: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.einsum("bij,bnj->bni", camera, joints)
    return homogeneous[..., :2] / homogeneous[..., 2:3].clamp(max=-1e-6)


def full_to_heatmap(xy_full: torch.Tensor, crop_to_full: torch.Tensor) -> torch.Tensor:
    inverse = torch.linalg.inv(crop_to_full)
    ones = torch.ones_like(xy_full[..., :1])
    homogeneous = torch.cat((xy_full, ones), dim=-1)
    low = torch.einsum("bij,bnj->bni", inverse, homogeneous)
    return low[..., :2] / low[..., 2:3]


def pose_energy_per_frame(observation: PoseObservationBatch, xy_full: torch.Tensor) -> torch.Tensor:
    xy = full_to_heatmap(xy_full, observation.crop_to_full)
    batch, joints, height, width = observation.heatmaps.shape
    x = 2.0 * xy[..., 0] / max(width - 1, 1) - 1.0
    y = 2.0 * xy[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1).reshape(batch * joints, 1, 1, 2)
    sampled = F.grid_sample(
        observation.heatmaps.reshape(batch * joints, 1, height, width), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(batch, joints)
    nll = -torch.log(sampled.clamp_min(1e-8))
    weight = observation.weight
    return (nll * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-8)


def pose_noise_sigma(observation: PoseObservationBatch, xy_full: torch.Tensor) -> torch.Tensor:
    # Propagate the detector's full-distribution covariance through four
    # deterministic +/- principal-axis probes.  This is observation-only.
    covariance = observation.covariance_full
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    shifts = []
    for axis in (0, 1):
        vector = eigenvectors[..., axis] * eigenvalues[..., axis].clamp_min(0).sqrt().unsqueeze(-1)
        shifts.extend((vector, -vector))
    energies = torch.stack([pose_energy_per_frame(observation, xy_full + shift) for shift in shifts])
    return energies.std(dim=0, unbiased=True).clamp_min(1e-4)


def _edge_indices(names: Sequence[str]) -> tuple[tuple[int, int], ...]:
    mapping = resolve_joint_indices(names, tuple({item for edge in NLF_EDGES_BY_NAME for item in edge}))
    return tuple((mapping[parent], mapping[child]) for parent, child in NLF_EDGES_BY_NAME)


def _bone_values(joints: torch.Tensor, valid: torch.Tensor, edges: tuple[tuple[int, int], ...]):
    parent = torch.as_tensor([edge[0] for edge in edges], device=joints.device)
    child = torch.as_tensor([edge[1] for edge in edges], device=joints.device)
    bone = joints[..., child, :] - joints[..., parent, :]
    length = torch.linalg.vector_norm(bone, dim=-1)
    usable = valid[..., child] & valid[..., parent] & (length > 1e-5)
    return bone / length.clamp_min(1e-8).unsqueeze(-1), length, usable


def nlf_energy_per_frame(
    observation: NLFObservationBatch,
    predicted_joints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = _edge_indices(observation.names)
    predicted_valid = torch.isfinite(predicted_joints).all(dim=-1)
    pred_unit, pred_length, pred_ok = _bone_values(predicted_joints, predicted_valid, edges)
    obs_unit_aug, obs_length_aug, obs_ok_aug = _bone_values(observation.joints, observation.valid, edges)
    obs_unit = F.normalize(obs_unit_aug.mean(dim=1), dim=-1)
    obs_length = obs_length_aug.median(dim=1).values
    obs_ok = obs_ok_aug.all(dim=1)
    # Robust per-frame scale from torso/shoulder bones; it is detached and no
    # rotation alignment is estimated.
    ratio = (pred_length / obs_length.clamp_min(1e-6)).detach()
    scale = ratio.median(dim=1).values.detach()
    residual = pred_unit - obs_unit
    absolute = residual.abs()
    direction = torch.where(absolute <= 0.05, 10.0 * residual.square(), absolute - 0.025).sum(-1)
    length = torch.abs(torch.log(pred_length / (scale[:, None] * obs_length).clamp_min(1e-6)))
    weight = (pred_ok & obs_ok).to(direction.dtype)
    energy = ((direction + 0.05 * length) * weight).sum(1) / weight.sum(1).clamp_min(1)
    # TTA noise: score every explicit augmentation against the ensemble target.
    aug_residual = obs_unit_aug - obs_unit[:, None]
    aug_abs = aug_residual.abs()
    aug_energy = torch.where(aug_abs <= 0.05, 10.0 * aug_residual.square(), aug_abs - 0.025).sum(-1)
    aug_weight = obs_ok_aug.to(aug_energy.dtype)
    aug_energy = (aug_energy * aug_weight).sum(-1) / aug_weight.sum(-1).clamp_min(1)
    sigma = aug_energy.std(dim=1, unbiased=True).clamp_min(1e-4)
    return energy, sigma


def _robust_scale(values: torch.Tensor) -> torch.Tensor:
    median = values.detach().median()
    mad = (values.detach() - median).abs().median()
    return mad.clamp_min(1e-3)


def _initial_projection_offset(
    pose: PoseObservationBatch,
    joints: torch.Tensor,
    cameras: torch.Tensor,
) -> torch.Tensor:
    smpl_mapping = resolve_joint_indices(__import__("smplx.joint_names", fromlist=["JOINT_NAMES"]).JOINT_NAMES, pose.names)
    indices = torch.as_tensor([smpl_mapping[name] for name in pose.names], device=joints.device)
    projected = project(joints.index_select(1, indices), cameras)
    # Distribution means are recoverable from heatmaps in low-res coordinates.
    h, w = pose.heatmaps.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=joints.device, dtype=torch.float32),
        torch.arange(w, device=joints.device, dtype=torch.float32), indexing="ij",
    )
    low_xy = torch.stack((xx.flatten(), yy.flatten()), -1)
    mean_low = pose.heatmaps.flatten(2) @ low_xy
    ones = torch.ones_like(mean_low[..., :1])
    mean_full_h = torch.einsum("bij,bnj->bni", pose.crop_to_full, torch.cat((mean_low, ones), -1))
    mean_full = mean_full_h[..., :2] / mean_full_h[..., 2:3]
    difference = mean_full - projected
    return difference.median(dim=1).values.detach()


def optimize_batch(
    records: Sequence[ManifestRecord],
    model_root: Path,
    pose_root: Path,
    nlf_root: Path | None,
    *,
    wrist_protection: bool,
    device: str,
    u1_steps: int,
    u2_steps: int,
    wrist_projection_steps: int = 5,
) -> tuple[TrustRegionSMPLX, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    baseline = BaselineBatch.from_npz([Path(item.a3f_state_path) for item in records], device)
    pose_observation = _load_pose(records, pose_root, device)
    nlf_observation = _load_nlf(records, nlf_root, device) if nlf_root is not None else None
    model = TrustRegionSMPLX(
        model_root, baseline, ALL_ACTIVE, wrist_protection=wrist_protection,
    )
    with torch.no_grad():
        baseline_output = model()
    pose_mapping = resolve_joint_indices(
        __import__("smplx.joint_names", fromlist=["JOINT_NAMES"]).JOINT_NAMES,
        pose_observation.names,
    )
    pose_indices = torch.as_tensor([pose_mapping[name] for name in pose_observation.names], device=device)
    nlf_indices = None
    if nlf_observation is not None:
        smpl_names = __import__("smplx.joint_names", fromlist=["JOINT_NAMES"]).JOINT_NAMES
        nlf_mapping = resolve_joint_indices(smpl_names, nlf_observation.names)
        nlf_indices = torch.as_tensor([nlf_mapping[name] for name in nlf_observation.names], device=device)
    offset0 = _initial_projection_offset(
        pose_observation, baseline_output["joints"], baseline.cameras,
    )
    offset_raw = torch.nn.Parameter(torch.zeros_like(offset0))

    def projection_offset() -> torch.Tensor:
        return offset0 + 20.0 * torch.tanh(offset_raw)

    with torch.no_grad():
        base_xy = project(baseline_output["joints"].index_select(1, pose_indices), baseline.cameras)
        base_pose = pose_energy_per_frame(pose_observation, base_xy + offset0[:, None])
        base_pose_sigma = pose_noise_sigma(pose_observation, base_xy + offset0[:, None])
        if nlf_observation is not None and nlf_indices is not None:
            base_nlf, base_nlf_sigma = nlf_energy_per_frame(
                nlf_observation, baseline_output["joints"].index_select(1, nlf_indices)
            )
        else:
            base_nlf = torch.zeros_like(base_pose)
            base_nlf_sigma = torch.full_like(base_pose, float("nan"))
    pose_scale = _robust_scale(base_pose)
    nlf_scale = _robust_scale(base_nlf) if nlf_observation is not None else torch.tensor(1.0, device=device)
    best_loss = torch.full_like(base_pose, float("inf"))
    best_delta = model.delta.detach().clone()
    best_offset_raw = offset_raw.detach().clone()

    stage_specs = ((U1_ACTIVE, u1_steps, 3e-3), (ALL_ACTIVE, u2_steps, 2e-3))
    active_lookup = {name: i for i, name in enumerate(ALL_ACTIVE)}
    for stage_names, steps, learning_rate in stage_specs:
        if steps <= 0:
            continue
        optimizer = torch.optim.Adam((model.delta, offset_raw), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
        active_parameter_indices = [active_lookup[name] for name in stage_names]
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output = model()
            xy = project(output["joints"].index_select(1, pose_indices), baseline.cameras)
            pose_value = pose_energy_per_frame(pose_observation, xy + projection_offset()[:, None])
            per_frame = (pose_value - base_pose.detach()) / pose_scale
            if nlf_observation is not None and nlf_indices is not None:
                nlf_value, _ = nlf_energy_per_frame(
                    nlf_observation, output["joints"].index_select(1, nlf_indices)
                )
                per_frame = per_frame + (nlf_value - base_nlf.detach()) / nlf_scale
            bounded = output["bounded_delta"]
            trust = (bounded / model.radii[None, :, None]).square().sum(-1).mean(-1)
            per_frame = per_frame + 0.5 * trust
            loss = per_frame.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite SignEFT objective")
            with torch.no_grad():
                improved = per_frame < best_loss
                best_loss[improved] = per_frame[improved]
                best_delta[improved] = model.delta.detach()[improved]
                best_offset_raw[improved] = offset_raw.detach()[improved]
            loss.backward()
            inactive = sorted(set(range(len(ALL_ACTIVE))) - set(active_parameter_indices))
            if inactive:
                model.delta.grad[:, inactive] = 0
            torch.nn.utils.clip_grad_norm_((model.delta, offset_raw), 1.0)
            optimizer.step()
            scheduler.step()
    with torch.no_grad():
        model.delta.copy_(best_delta)
        offset_raw.copy_(best_offset_raw)
    if wrist_protection and wrist_projection_steps > 0:
        left_ids, right_ids = load_mano_vertex_ids(model_root)
        left_tensor = torch.as_tensor(left_ids, device=device)
        right_tensor = torch.as_tensor(right_ids, device=device)
        model.wrist_projection_delta.requires_grad_(True)
        projection_optimizer = torch.optim.Adam((model.wrist_projection_delta,), lr=1e-3)
        best_projection_loss = torch.full_like(base_pose, float("inf"))
        best_projection_delta = model.wrist_projection_delta.detach().clone()
        for _ in range(wrist_projection_steps):
            projection_optimizer.zero_grad(set_to_none=True)
            projected_output = model()
            left = translation_aligned_hand_distance(
                projected_output["vertices"].index_select(1, left_tensor),
                baseline.cached_vertices.index_select(1, left_tensor),
            )
            right = translation_aligned_hand_distance(
                projected_output["vertices"].index_select(1, right_tensor),
                baseline.cached_vertices.index_select(1, right_tensor),
            )
            per_frame_projection = left + right
            with torch.no_grad():
                improved = per_frame_projection < best_projection_loss
                best_projection_loss[improved] = per_frame_projection[improved]
                best_projection_delta[improved] = model.wrist_projection_delta.detach()[improved]
            per_frame_projection.mean().backward()
            torch.nn.utils.clip_grad_norm_((model.wrist_projection_delta,), 0.1)
            projection_optimizer.step()
        with torch.no_grad():
            model.wrist_projection_delta.copy_(best_projection_delta)
        model.wrist_projection_delta.requires_grad_(False)
    with torch.no_grad():
        candidate = model()
        candidate_xy = project(candidate["joints"].index_select(1, pose_indices), baseline.cameras)
        candidate_pose = pose_energy_per_frame(
            pose_observation, candidate_xy + projection_offset()[:, None]
        )
        if nlf_observation is not None and nlf_indices is not None:
            candidate_nlf, _ = nlf_energy_per_frame(
                nlf_observation, candidate["joints"].index_select(1, nlf_indices)
            )
        else:
            candidate_nlf = torch.zeros_like(candidate_pose)
    evidence = {
        "base_pose": base_pose, "candidate_pose": candidate_pose, "pose_sigma": base_pose_sigma,
        "base_nlf": base_nlf, "candidate_nlf": candidate_nlf, "nlf_sigma": base_nlf_sigma,
        "projection_offset": projection_offset().detach(), "best_loss": best_loss,
        "wrist_projection_delta_deg": torch.rad2deg(
            torch.linalg.vector_norm(model.wrist_projection_delta.detach(), dim=-1)
        ),
    }
    candidate["baseline_vertices"] = baseline.cached_vertices
    candidate["baseline_joints"] = baseline_output["joints"]
    candidate["cameras"] = baseline.cameras
    return model, candidate, evidence


def run_core_refinement(
    manifest_path: Path,
    run_root: Path,
    model_root: Path,
    pose_root: Path,
    nlf_root: Path | None,
    *,
    wrist_protection: bool,
    device: str = "cuda",
    batch_size: int = 8,
    u1_steps: int = 50,
    u2_steps: int = 75,
    wrist_projection_steps: int = 5,
    limit: int | None = None,
) -> dict[str, object]:
    all_records = read_manifest(manifest_path)
    records = all_records if limit is None else all_records[:limit]
    accepted_count = 0
    reason_histogram: dict[str, int] = {}
    implementation_sha = tree_sha256(Path(__file__).parents[1])
    pending = []
    resumed = 0
    for record in records:
        output_obj = run_root / "frames" / record.sign_id / f"{record.source_frame_id:06d}.obj"
        output_state = run_root / "frames" / record.sign_id / f"{record.source_frame_id:06d}.npz"
        decision_path = run_root / "decisions" / record.sign_id / f"{record.source_frame_id:06d}.json"
        existing = (output_obj.is_file(), output_state.is_file(), decision_path.is_file())
        if any(existing) and not all(existing):
            raise RuntimeError(f"incomplete resumable refinement output: {record.record_id}")
        if all(existing):
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if decision["record_id"] != record.record_id:
                raise RuntimeError(f"resumable decision identity mismatch: {record.record_id}")
            if decision.get("implementation_sha256") != implementation_sha:
                raise RuntimeError(f"resumable implementation drift: {record.record_id}")
            pose_path = pose_root / record.sign_id / f"{record.source_frame_id:06d}.npz"
            if decision["input_hashes"].get("pose_observation") != sha256_file(pose_path):
                raise RuntimeError(f"resumable pose observation drift: {record.record_id}")
            if nlf_root is not None:
                nlf_path = nlf_root / record.sign_id / f"{record.source_frame_id:06d}.npz"
                if decision["input_hashes"].get("nlf_observation") != sha256_file(nlf_path):
                    raise RuntimeError(f"resumable NLF observation drift: {record.record_id}")
            if decision["output_hashes"]["obj"] != sha256_file(output_obj):
                raise RuntimeError(f"resumable OBJ hash mismatch: {record.record_id}")
            if decision["output_hashes"]["state"] != sha256_file(output_state):
                raise RuntimeError(f"resumable state hash mismatch: {record.record_id}")
            accepted_count += int(decision["accepted"])
            reason = str(decision["reason"])
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
            resumed += 1
        else:
            pending.append(record)
    for start in range(0, len(pending), batch_size):
        batch_records = pending[start:start + batch_size]
        model, candidate, evidence = optimize_batch(
            batch_records, model_root, pose_root, nlf_root,
            wrist_protection=wrist_protection, device=device,
            u1_steps=u1_steps, u2_steps=u2_steps,
            wrist_projection_steps=wrist_projection_steps,
        )
        left_ids, right_ids = load_mano_vertex_ids(model_root)
        left_tensor = torch.as_tensor(left_ids, device=device)
        right_tensor = torch.as_tensor(right_ids, device=device)
        face_tensor, lower_tensor = semantic_vertex_ids(model.model)
        left_drift = translation_aligned_hand_distance(
            candidate["vertices"].index_select(1, left_tensor),
            candidate["baseline_vertices"].index_select(1, left_tensor),
        ) * 1000.0
        right_drift = translation_aligned_hand_distance(
            candidate["vertices"].index_select(1, right_tensor),
            candidate["baseline_vertices"].index_select(1, right_tensor),
        ) * 1000.0
        vertex_displacement_mm = torch.linalg.vector_norm(
            candidate["vertices"] - candidate["baseline_vertices"], dim=-1
        ) * 1000.0
        face_drift = vertex_displacement_mm.index_select(1, face_tensor).mean(1)
        lower_drift = vertex_displacement_mm.index_select(1, lower_tensor).mean(1)
        candidate_globals = model._global_rotations(model.root_R0, candidate["body_rotations"])
        wrist_relative = torch.stack((
            candidate_globals[:, 20] @ model.wrist_global_left0.transpose(-1, -2),
            candidate_globals[:, 21] @ model.wrist_global_right0.transpose(-1, -2),
        ), dim=1)
        wrist_cosine = (
            (torch.diagonal(wrist_relative, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        wrist_global_deviation_deg = torch.rad2deg(torch.acos(wrist_cosine))
        bounded_deg = torch.rad2deg(torch.linalg.vector_norm(candidate["bounded_delta"], dim=-1))
        for i, record in enumerate(batch_records):
            delta_pose = float(evidence["candidate_pose"][i] - evidence["base_pose"][i])
            families = [FamilyDelta("pose2d", delta_pose, float(evidence["pose_sigma"][i]))]
            delta_nlf = None
            if nlf_root is not None:
                delta_nlf = float(evidence["candidate_nlf"][i] - evidence["base_nlf"][i])
                families.append(FamilyDelta("nlf3d", delta_nlf, float(evidence["nlf_sigma"][i])))
            finite = bool(torch.isfinite(candidate["vertices"][i]).all())
            hand_ok = float(left_drift[i]) <= 0.25 and float(right_drift[i]) <= 0.25
            off_target_ok = float(face_drift[i]) <= 2.0 and float(lower_drift[i]) <= 0.25
            wrist_orientation_ok = bool((wrist_global_deviation_deg[i] <= 1.01).all())
            geometry_ok = finite and hand_ok and off_target_ok and wrist_orientation_ok
            radius_deg = torch.rad2deg(model.radii)
            trust_ok = bool((bounded_deg[i] <= radius_deg + 1e-4).all())
            bound_hits = float((bounded_deg[i] >= 0.99 * radius_deg).float().mean())
            trust_ok = trust_ok and bound_hits <= 0.10
            # Depth flag is based on decoded joint movement, not the residual parameter.
            changes_depth = bool(
                (candidate["joints"][i, :22, 2] - candidate["baseline_joints"][i, :22, 2]).abs().max() > 5e-4
            )
            accepted, winners, losers, reason = accept_ubody(
                families, changes_depth=changes_depth, geometry_ok=geometry_ok, trust_ok=trust_ok,
            )
            if not wrist_protection and not hand_ok:
                reason = "HAND_PROTECTION_REQUIRED"
                accepted = False
            output_obj = run_root / "frames" / record.sign_id / f"{record.source_frame_id:06d}.obj"
            output_state = run_root / "frames" / record.sign_id / f"{record.source_frame_id:06d}.npz"
            if accepted:
                _, faces = load_obj(Path(record.a3f_obj_path))
                vertices = candidate["vertices"][i].detach().cpu().numpy().astype(np.float32)
                write_obj(output_obj, vertices, faces)
                with np.load(record.a3f_state_path, allow_pickle=False) as baseline_archive:
                    state = {key: np.asarray(baseline_archive[key]).copy() for key in baseline_archive.files}
                state["body_pose"] = candidate["body_axis_angle"][i:i + 1].detach().cpu().numpy().reshape(1, 63).astype(np.float32)
                state["vertices"] = vertices
                atomic_savez(output_state, **state)
                accepted_count += 1
            else:
                exact_rollback(
                    Path(record.a3f_obj_path), Path(record.a3f_state_path), output_obj, output_state,
                )
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
            decision = {
                "schema_version": "signeft.decision.v1",
                "implementation_sha256": implementation_sha,
                "record_id": record.record_id,
                "candidate_id": "C3" if wrist_protection else ("C2" if nlf_root else "C1"),
                "accepted": accepted,
                "winning_families": winners,
                "losing_families": losers,
                "energy_delta": {"pose2d": delta_pose, "nlf3d": delta_nlf},
                "noise_sigma": {
                    "pose2d": float(evidence["pose_sigma"][i]),
                    "nlf3d": float(evidence["nlf_sigma"][i]) if nlf_root is not None else None,
                },
                "trust_max_deg": {
                    name: float(value) for name, value in zip(ALL_ACTIVE, bounded_deg[i])
                },
                "bound_hit_fraction": bound_hits,
                "lhand_centered_drift_mm": float(left_drift[i]),
                "rhand_centered_drift_mm": float(right_drift[i]),
                "off_target_drift_mm": {
                    "face_mean": float(face_drift[i]),
                    "lower_body_mean": float(lower_drift[i]),
                },
                "fallback": None if accepted else "exact_a3f",
                "reason": reason,
                "changes_depth": changes_depth,
                "projection_offset_px": evidence["projection_offset"][i].cpu().tolist(),
                "wrist_projection_delta_deg": evidence["wrist_projection_delta_deg"][i].cpu().tolist(),
                "wrist_global_deviation_deg": wrist_global_deviation_deg[i].detach().cpu().tolist(),
                "input_hashes": {
                    "rgb": record.sha256_rgb, "obj": record.sha256_a3f_obj,
                    "state": record.sha256_a3f_state,
                    "pose_observation": sha256_file(
                        pose_root / record.sign_id / f"{record.source_frame_id:06d}.npz"
                    ),
                    "nlf_observation": sha256_file(
                        nlf_root / record.sign_id / f"{record.source_frame_id:06d}.npz"
                    ) if nlf_root is not None else None,
                },
                "output_hashes": {"obj": sha256_file(output_obj), "state": sha256_file(output_state)},
                "objective_uses_ground_truth": False,
                "objective_uses_temporal_pose": False,
            }
            atomic_write_json(
                run_root / "decisions" / record.sign_id / f"{record.source_frame_id:06d}.json",
                decision,
            )
    summary = {
        "schema_version": "signeft.refinement-summary.v1",
        "status": "ok",
        "frames": len(records),
        "manifest_frames": len(all_records),
        "accepted": accepted_count,
        "fallback": len(records) - accepted_count,
        "acceptance_rate": accepted_count / max(len(records), 1),
        "resumed": resumed,
        "written": len(pending),
        "reason_histogram": reason_histogram,
        "wrist_protection": wrist_protection,
        "nlf_enabled": nlf_root is not None,
        "objective_uses_ground_truth": False,
        "objective_uses_temporal_pose": False,
        "implementation_sha256": implementation_sha,
    }
    atomic_write_json(run_root / "refinement_summary.json", summary)
    return summary
