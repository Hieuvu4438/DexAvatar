from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from signpccx.data.manifest import FrameRecord, read_jsonl
from signpccx.geometry.contact_regions import nearest_region
from signpccx.io import atomic_write_json, atomic_write_text, sha256_file
from signpccx.model.canonicalizer import load_mano_smplx_ids
from signpccx.model.smplx_state import (
    ARM_SLOTS,
    UPPER_BODY_SLOTS,
    WRIST_SLOT,
    FrameState,
    SharedSignedCamera,
    mask_body_gradient,
    validate_body_slots,
)
from signpccx.optimization.contact import ContactProposal, gated_contact_loss
from signpccx.optimization.hypotheses import HandHypothesis, chirality_loss, twist_wrist
from signpccx.optimization.identity import farthest_point_indices, huber_location
from signpccx.teachers.observations import (
    BODY_ANCHORS,
    LEFT_ANCHORS,
    RIGHT_ANCHORS,
    FrameObservation,
    load_frame_observation,
    observation_initial_state,
)


BODY_JOINT_IDS = (9, 12, 16, 17, 18, 19, 20, 21)
CHAIN_IDS = {"left": (16, 18, 20), "right": (17, 19, 21)}
HAND_JOINT_IDS = {
    "left": (20, 37, 25, 28, 34, 31, 66, 67, 68, 69, 70),
    "right": (21, 52, 40, 43, 49, 46, 71, 72, 73, 74, 75),
}
HAND_ANCHORS = {"left": LEFT_ANCHORS, "right": RIGHT_ANCHORS}
PALM_CHIRALITY_IDS = (0, 2, 5)  # wrist, index MCP, pinky MCP in hand subset
S3_BODY_SLOTS = (15, 16, 17, 18, 19, 20)


@dataclass(frozen=True)
class GeometryRegions:
    indices: dict[str, object]
    surface_faces: dict[str, object]
    upper_vertices: object
    face_vertices: object
    vertex_weights: object


@dataclass
class FitContext:
    observation: FrameObservation
    model: object
    beta: object
    camera: SharedSignedCamera | None
    regions: GeometryRegions
    faces: object
    initial: FrameState
    weights: dict[str, float]
    contact_proposals: list[ContactProposal]
    device: str


@dataclass(frozen=True)
class CandidateResult:
    side: str
    name: str
    state: FrameState
    score: float
    components: dict[str, float]


def _tensor(value, device: str):
    import torch

    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _full_intrinsics(observation: FrameObservation) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2] = observation.arrays["image_to_crop"]
    return (np.linalg.inv(homography) @ observation.arrays["K_crop"]).astype(np.float32)


def _camera_for_context(context: FitContext):
    from signpccx.optimization.losses import affine_homogeneous

    if context.camera is None:
        return _tensor(context.observation.arrays["K_crop"], context.device).unsqueeze(0)
    transform = _tensor(context.observation.arrays["image_to_crop"], context.device)
    return affine_homogeneous(transform) @ context.camera.matrix().unsqueeze(0)


def _anchor_joints(joints):
    import torch

    chest = (joints[:, 16:17] + joints[:, 17:18]) * 0.5
    body = torch.cat((chest, joints[:, [12, 16, 17, 18, 19, 20, 21]]), dim=1)
    left_ids = list(HAND_JOINT_IDS["left"])
    right_ids = list(HAND_JOINT_IDS["right"])
    left = joints[:, left_ids]
    right = joints[:, right_ids]
    left_palm = left[:, 1:6].mean(dim=1, keepdim=True)
    right_palm = right[:, 1:6].mean(dim=1, keepdim=True)
    return torch.cat((body, left_palm, left[:, 1:], right_palm, right[:, 1:]), dim=1)


def _surface_faces(faces, allowed_ids, vertex_count: int):
    import torch

    mask = torch.zeros(vertex_count, dtype=torch.bool, device=faces.device)
    mask[allowed_ids] = True
    selected = faces[mask[faces].all(dim=1)]
    if not len(selected):
        raise RuntimeError("anatomical region produced no complete faces")
    return selected


def build_geometry_regions(model, mano_ids_path: Path, device: str, vertices_per_tip: int = 24) -> GeometryRegions:
    import torch

    left_np, right_np = load_mano_smplx_ids(mano_ids_path)
    left = torch.as_tensor(left_np, dtype=torch.long, device=device)
    right = torch.as_tensor(right_np, dtype=torch.long, device=device)
    with torch.no_grad():
        neutral = model(return_verts=True)
        vertices = neutral.vertices[0].detach().cpu().numpy()
        joints = neutral.joints[0].detach().cpu().numpy()
        lbs = model.lbs_weights.detach()
    tip_joint_ids = {"left": (66, 67, 68, 69, 70), "right": (71, 72, 73, 74, 75)}
    allowed = {"left": left_np, "right": right_np}
    regions: dict[str, torch.Tensor] = {"left_hand": left, "right_hand": right}
    for side in ("left", "right"):
        tips = [nearest_region(vertices, joints[joint], allowed[side], vertices_per_tip) for joint in tip_joint_ids[side]]
        regions[f"{side}_fingertips"] = torch.as_tensor(
            np.unique(np.concatenate(tips)), dtype=torch.long, device=device
        )
    upper_weight = lbs[:, [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]].sum(dim=1)
    face_weight = lbs[:, [15, 22, 23, 24]].sum(dim=1)
    torso_weight = lbs[:, [3, 6, 9, 12, 13, 14]].sum(dim=1)
    upper = torch.where(upper_weight > 0.30)[0]
    face = torch.where(face_weight > 0.20)[0]
    torso = torch.where(torso_weight > 0.30)[0]
    regions["face"] = face
    regions["upper_torso"] = torso
    faces = model.faces_tensor.to(device=device, dtype=torch.long)
    surfaces = {
        "left_hand": _surface_faces(faces, left, len(vertices)),
        "right_hand": _surface_faces(faces, right, len(vertices)),
        "face": _surface_faces(faces, face, len(vertices)),
        "upper_torso": _surface_faces(faces, torso, len(vertices)),
    }
    vertex_weights = torch.ones(len(vertices), dtype=torch.float32, device=device)
    vertex_weights[upper] = 2.0
    vertex_weights[face] = 0.3
    vertex_weights[left] = 3.0
    vertex_weights[right] = 3.0
    # Boundary vertices get the blueprint's lower seam weight.
    hand_mask = torch.zeros(len(vertices), dtype=torch.bool, device=device)
    hand_mask[left] = True
    hand_mask[right] = True
    crossing = faces[hand_mask[faces].any(dim=1) & ~hand_mask[faces].all(dim=1)]
    seam = torch.unique(crossing[hand_mask[crossing]])
    vertex_weights[seam] = 0.5
    return GeometryRegions(regions, surfaces, upper, face, vertex_weights)


def _transform_points(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    return (affine @ homogeneous.T).T


def _proposal_confidence(distance_px: float, confidence: float, radius_px: float) -> float:
    return float(np.clip(confidence * (1.0 - distance_px / radius_px), 0.0, 1.0))


def build_contact_proposals(observation: FrameObservation, threshold: float = 0.70) -> list[ContactProposal]:
    a = observation.arrays
    uv = a["anchor_uv_observed"]
    conf = a["anchor_uv_confidence"] * a["anchor_valid"].astype(np.float32)
    proposals: list[ContactProposal] = []

    def closest(first: Iterable[int], second_points: np.ndarray, second_conf: np.ndarray):
        first_ids = np.asarray(tuple(first), dtype=np.int64)
        valid_first = conf[first_ids] > 0
        valid_second = second_conf > 0
        if not valid_first.any() or not valid_second.any():
            return math.inf, 0.0
        distances = np.linalg.norm(
            uv[first_ids][valid_first, None] - second_points[valid_second][None], axis=-1
        )
        index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        c1 = conf[first_ids][valid_first][index[0]]
        c2 = second_conf[valid_second][index[1]]
        return float(distances[index]), float(min(c1, c2))

    hand_specs = (("left", range(14, 19), "right", range(19, 30)),
                  ("right", range(25, 30), "left", range(8, 19)))
    for source, source_ids, target, target_ids in hand_specs:
        target_ids = np.asarray(tuple(target_ids), dtype=np.int64)
        distance, confidence = closest(source_ids, uv[target_ids], conf[target_ids])
        score = _proposal_confidence(distance, confidence, 30.0)
        if score >= threshold:
            proposals.append(ContactProposal(
                f"{source}_fingertips", f"{target}_hand", score, 0.003
            ))

    dwpose = a["dwpose_keypoints"]
    face_full = dwpose[24:92, :2]
    face_conf = dwpose[24:92, 2]
    face_crop = _transform_points(a["image_to_crop"], face_full)
    torso_ids = np.asarray((0, 1, 2, 3), dtype=np.int64)
    for side, tip_ids in (("left", range(14, 19)), ("right", range(25, 30))):
        distance, confidence = closest(tip_ids, face_crop, face_conf)
        score = _proposal_confidence(distance, confidence, 24.0)
        if score >= threshold:
            proposals.append(ContactProposal(f"{side}_fingertips", "face", score, 0.006))
        distance, confidence = closest(tip_ids, uv[torso_ids], conf[torso_ids])
        score = _proposal_confidence(distance, confidence, 30.0)
        if score >= threshold:
            proposals.append(ContactProposal(f"{side}_hand", "upper_torso", score, 0.006))
    return proposals


def _raw_losses(
    context: FitContext,
    state: FrameState,
    needed: set[str],
) -> tuple[dict[str, object], object]:
    import torch
    from signpccx.optimization.losses import (
        arm_chain_loss,
        centered_point_loss,
        keypoint_loss,
        penetration_from_surfaces,
        pose_anatomy_loss,
        safe_project,
    )

    output = context.model(**state.smplx_kwargs(context.beta))
    anchors = _anchor_joints(output.joints)
    predicted_uv = safe_project(anchors, _camera_for_context(context))
    a = context.observation.arrays
    observed_uv = _tensor(a["anchor_uv_observed"], context.device).unsqueeze(0)
    confidence = _tensor(
        a["anchor_uv_confidence"] * a["anchor_valid"].astype(np.float32), context.device
    ).unsqueeze(0)
    target_anchor = _tensor(a["init_anchor_cam"], context.device).unsqueeze(0)
    part = torch.ones(30, dtype=torch.float32, device=context.device)
    part[:4] = 1.5
    part[4:8] = 2.0
    part[8:] = 2.5
    losses: dict[str, object] = {}
    if "body_2d" in needed:
        losses["body_2d"] = keypoint_loss(
            predicted_uv[:, :8], observed_uv[:, :8], confidence[:, :8],
            context.observation.crop_hw, part[:8],
        )
    if "hand_2d" in needed:
        losses["hand_2d"] = keypoint_loss(
            predicted_uv[:, 8:], observed_uv[:, 8:], confidence[:, 8:],
            context.observation.crop_hw, part[8:],
        )
    target_joints = _tensor(a["smplx_joints_parametric"], context.device).unsqueeze(0)
    if "body_teacher" in needed:
        joint_term = centered_point_loss(
            output.joints[:, list(BODY_JOINT_IDS)], target_joints[:, list(BODY_JOINT_IDS)]
        )
        dense_term = centered_point_loss(
            output.vertices[:, context.regions.upper_vertices],
            _tensor(a["mesh_hybrid_init"], context.device).unsqueeze(0)[:, context.regions.upper_vertices],
        )
        losses["body_teacher"] = 0.5 * (joint_term + dense_term)
    if "hand_teacher" in needed:
        hand_terms = []
        for side in ("left", "right"):
            ids = list(HAND_ANCHORS[side])
            hand_terms.append(centered_point_loss(anchors[:, ids], target_anchor[:, ids], confidence[:, ids]))
        losses["hand_teacher"] = 0.5 * sum(hand_terms)
    if "arm_chain" in needed:
        chain_terms = []
        for side in ("left", "right"):
            source_ids = CHAIN_IDS[side]
            chain_conf = confidence[:, 2:8].mean(dim=1)
            chain_terms.append(arm_chain_loss(output.joints, target_joints, source_ids, chain_conf))
        losses["arm_chain"] = 0.5 * sum(chain_terms)
    if "palm_chirality" in needed:
        palm_terms = []
        crop_diagonal = float(sum(value * value for value in context.observation.crop_hw)) ** 0.5
        for side in ("left", "right"):
            ids = list(HAND_ANCHORS[side])
            palm_terms.append(chirality_loss(
                predicted_uv[:, ids] / crop_diagonal,
                observed_uv[:, ids] / crop_diagonal,
                PALM_CHIRALITY_IDS,
            ))
        losses["palm_chirality"] = 0.5 * sum(palm_terms)
    if "anatomy" in needed:
        losses["anatomy"] = pose_anatomy_loss(
            state.body_pose, state.left_hand_pose, state.right_hand_pose
        )
    if "pose_anchor" in needed:
        losses["pose_anchor"] = sum(
            (getattr(state, name) - getattr(context.initial, name)).square().mean()
            for name in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")
        )
    active = 0
    if "contact" in needed:
        contact, active = gated_contact_loss(
            output.vertices, context.contact_proposals, context.regions.indices
        )
        losses["contact"] = contact
    if "penetration" in needed:
        penetration_terms = []
        max_depth = output.vertices.new_zeros(())
        collision_count = output.vertices.new_zeros((), dtype=torch.long)
        # Symmetric, oriented point-to-triangle hand penetration. Query sampling is
        # deterministic; closest-triangle discovery is no_grad inside the loss.
        for query, surface in (("left_hand", "right_hand"), ("right_hand", "left_hand")):
            query_ids = context.regions.indices[query][::8]
            term, depth, count = penetration_from_surfaces(
                output.vertices, query_ids, context.regions.surface_faces[surface]
            )
            penetration_terms.append(term)
            max_depth = torch.maximum(max_depth, depth)
            collision_count = collision_count + count
        for proposal in context.contact_proposals:
            if proposal.region_b not in ("face", "upper_torso"):
                continue
            query_ids = context.regions.indices[proposal.region_a][::4]
            term, depth, count = penetration_from_surfaces(
                output.vertices, query_ids, context.regions.surface_faces[proposal.region_b]
            )
            penetration_terms.append(term)
            max_depth = torch.maximum(max_depth, depth)
            collision_count = collision_count + count
        losses["penetration"] = sum(penetration_terms) / len(penetration_terms)
        losses["penetration_depth"] = max_depth
        losses["penetration_count"] = collision_count
    if "shape_prior" in needed:
        losses["shape_prior"] = context.beta.square().mean()
    if "root_anchor" in needed:
        losses["root_anchor"] = (state.transl - context.initial.transl).square().mean()
    if "contact" in needed:
        losses["contact_active"] = active
    return losses, output


def _objective(context: FitContext, state: FrameState, enabled: set[str]):
    raw, output = _raw_losses(context, state, enabled)
    weighted = {}
    total = output.vertices.new_zeros(())
    for name in enabled:
        value = raw[name]
        coefficient = context.weights.get(name, 1.0)
        weighted[name] = value * coefficient
        total = total + weighted[name]
    return total, raw, weighted, output


def _write_step(handle, stage: str, step: int, total, raw: dict, weighted: dict, grad_norm: float, **extra) -> None:
    import torch

    def scalar(value):
        if isinstance(value, (int, float)):
            return value
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    record = {
        "stage": stage, "step": step, "total": scalar(total), "finite": bool(torch.isfinite(total)),
        "grad_norm": float(grad_norm),
        "raw": {name: scalar(value) for name, value in raw.items()},
        "weighted": {name: scalar(value) for name, value in weighted.items()},
        **extra,
    }
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def optimize_adam(
    context: FitContext,
    state: FrameState,
    parameters: list,
    enabled: set[str],
    steps: int,
    learning_rate: float,
    active_body_slots: tuple[int, ...],
    stage: str,
    log_handle,
    grad_clip: float = 5.0,
    patience: int = 20,
    minimum_steps: int = 0,
) -> tuple[float, int]:
    import torch

    steps = int(steps)
    minimum_steps = int(minimum_steps)
    if not 0 <= minimum_steps <= steps:
        raise ValueError(
            f"{stage}: minimum_steps={minimum_steps} must be within [0, {steps}]"
        )
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    best = state.snapshot()
    best_loss = math.inf
    stale = 0
    completed = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, raw, weighted, _ = _objective(context, state, enabled)
        if not torch.isfinite(total):
            state.restore(best)
            raise FloatingPointError(f"{stage}: non-finite loss at step {step}")
        total.backward()
        mask_body_gradient(state, active_body_slots)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
        optimizer.step()
        current = float(total.detach())
        _write_step(log_handle, stage, step, total, raw, weighted, float(grad_norm))
        completed += 1
        if current + 1e-8 < best_loss:
            best_loss = current
            best = state.snapshot()
            stale = 0
        else:
            stale += 1
        if completed >= minimum_steps and stale >= int(patience):
            break
    state.restore(best)
    return best_loss, completed


def optimize_lbfgs(
    context: FitContext,
    state: FrameState,
    parameters: list,
    enabled: set[str],
    steps: int,
    learning_rate: float,
    active_body_slots: tuple[int, ...],
    log_handle,
) -> tuple[bool, float]:
    import torch

    snapshot = state.snapshot()
    with torch.no_grad():
        before = float(_objective(context, state, enabled)[0])
    optimizer = torch.optim.LBFGS(
        parameters, lr=float(learning_rate), max_iter=int(steps),
        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
    )
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        total, raw, weighted, _ = _objective(context, state, enabled)
        if not torch.isfinite(total):
            raise FloatingPointError("S4_lbfgs_refine: non-finite closure")
        total.backward()
        mask_body_gradient(state, active_body_slots)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        _write_step(log_handle, "S4_lbfgs_refine", calls, total, raw, weighted, float(grad_norm))
        calls += 1
        return total

    try:
        optimizer.step(closure)
        with torch.no_grad():
            after = float(_objective(context, state, enabled)[0])
        accepted = math.isfinite(after) and after <= before + 1e-8
    except (FloatingPointError, RuntimeError):
        accepted, after = False, math.inf
    if not accepted:
        state.restore(snapshot)
        after = before
    log_handle.write(json.dumps({
        "stage": "S4_lbfgs_refine", "event": "decision", "before": before,
        "after": after, "lbfgs_rejected": not accepted,
        "configured_max_iter": int(steps),
    }, sort_keys=True) + "\n")
    log_handle.flush()
    return accepted, after


def _candidate_sources(context: FitContext, side: str, degrees: tuple[float, ...]) -> list[HandHypothesis]:
    import torch

    a = context.observation.arrays
    wrist = context.initial.body_pose.detach()[0, WRIST_SLOT[side]].clone()
    fingers = _tensor(a[f"wilor_{side}_hand_pose_aa"], context.device).reshape(15, 3)
    base = HandHypothesis("h4w_cham_wilor", wrist, fingers, "H4WPP_CHAM+WiLoR")
    candidates = [base]
    if context.observation.smplerx is not None:
        smplerx = context.observation.smplerx
        smplerx_wrist = _tensor(smplerx["body_pose"], context.device).reshape(21, 3)[WRIST_SLOT[side]]
        candidates.append(HandHypothesis("smplerx_wrist_wilor", smplerx_wrist, fingers, "SMPLer-X+WiLoR"))
    neutral = context.model(return_verts=True).joints[0].detach()
    wrist_joint = HAND_JOINT_IDS[side][0]
    middle_mcp = HAND_JOINT_IDS[side][3]
    local_axis = neutral[middle_mcp] - neutral[wrist_joint]
    for degree in degrees:
        if float(degree) == 0.0:
            continue
        candidates.append(HandHypothesis(
            f"h4w_twist_{degree:+g}", twist_wrist(wrist, local_axis, degree), fingers,
            "H4WPP_CHAM_twist+WiLoR",
        ))
    hamer = context.observation.hamer_fingers.get(side)
    if hamer is not None:
        hamer_tensor = _tensor(hamer, context.device).reshape(15, 3)
        if float(torch.linalg.vector_norm(hamer_tensor - fingers, dim=-1).mean()) > 0.02:
            candidates.append(HandHypothesis("h4w_wrist_hamer", wrist, hamer_tensor, "H4WPP_CHAM+HaMeR"))
    return candidates


def _apply_candidate(state: FrameState, candidate: HandHypothesis, side: str) -> None:
    import torch

    with torch.no_grad():
        state.body_pose[0, WRIST_SLOT[side]].copy_(candidate.wrist_axis_angle)
        getattr(state, f"{side}_hand_pose")[0].copy_(candidate.hand_pose)


def _score(context: FitContext, state: FrameState, enabled: set[str]) -> tuple[float, dict[str, float]]:
    import torch

    with torch.no_grad():
        total, raw, _, _ = _objective(context, state, enabled)
    components = {
        key: float(value.detach().cpu()) for key, value in raw.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
    return float(total.detach().cpu()), components


def _rank_k0(context: FitContext, body_state: FrameState, side: str, degrees, keep: int, log_handle):
    enabled = {"hand_2d", "hand_teacher", "arm_chain", "palm_chirality", "anatomy"}
    scored = []
    for candidate in _candidate_sources(context, side, degrees):
        state = body_state.clone()
        _apply_candidate(state, candidate, side)
        score, components = _score(context, state, enabled)
        if math.isfinite(score):
            scored.append(CandidateResult(side, candidate.name, state, score, components))
        log_handle.write(json.dumps({
            "stage": "K0", "side": side, "candidate": candidate.name,
            "source": candidate.source, "score": score, "components": components,
            "finite": math.isfinite(score),
        }, sort_keys=True) + "\n")
    log_handle.flush()
    return sorted(scored, key=lambda result: (
        result.score, result.components.get("penetration", 0.0),
        result.components.get("hand_teacher", 0.0), result.components.get("anatomy", 0.0), result.name,
    ))[:keep]


def _fit_k1(context: FitContext, candidates, side: str, cfg: dict, log_handle):
    results = []
    stage_cfg = cfg["optimization"]["stages"]["hand_candidate"]
    enabled = {"hand_2d", "hand_teacher", "arm_chain", "palm_chirality", "anatomy", "pose_anchor"}
    for candidate in candidates:
        state = candidate.state.clone()
        parameters = [state.body_pose, getattr(state, f"{side}_hand_pose")]
        optimize_adam(
            context, state, parameters, enabled, stage_cfg["steps"], stage_cfg["lr"],
            (WRIST_SLOT[side],), f"S2_hand_hypothesis_{side}_{candidate.name}", log_handle,
            cfg["optimization"].get("grad_clip", 5.0),
            cfg["optimization"].get("early_stop_patience", 20),
            stage_cfg.get("min_steps", 25),
        )
        score, components = _score(context, state, enabled)
        results.append(CandidateResult(side, candidate.name, state, score, components))
    return sorted(results, key=lambda result: (
        result.score, result.components.get("penetration", 0.0),
        result.components.get("hand_teacher", 0.0), result.components.get("anatomy", 0.0), result.name,
    ))[: int(cfg["hypotheses"].get("fine_keep_per_hand", 2))]


def _combine_pair(left: CandidateResult, right: CandidateResult) -> FrameState:
    import torch

    state = left.state.clone()
    with torch.no_grad():
        state.body_pose[0, WRIST_SLOT["right"]].copy_(right.state.body_pose[0, WRIST_SLOT["right"]])
        state.right_hand_pose.copy_(right.state.right_hand_pose)
    return state


def _canonical_refit(context: FitContext, state: FrameState, cfg: dict, log_handle) -> tuple[FrameState, dict[str, float]]:
    import torch
    from signpccx.optimization.losses import centered_point_loss

    target_vertices = _tensor(context.observation.arrays["mesh_hybrid_init"], context.device).unsqueeze(0)
    with torch.no_grad():
        fitted_joints = context.model(**state.smplx_kwargs(context.beta)).joints.detach()
    canonical = state.clone()
    stage_cfg = cfg["optimization"]["stages"]["canonical"]
    minimum_steps = int(stage_cfg.get("min_steps", 20))
    if not 0 <= minimum_steps <= int(stage_cfg["steps"]):
        raise ValueError("S5_canonical_refit minimum_steps outside configured step budget")
    optimizer = torch.optim.Adam(
        [canonical.global_orient, canonical.body_pose, canonical.left_hand_pose,
         canonical.right_hand_pose, canonical.expression, canonical.transl],
        lr=float(stage_cfg["lr"]),
    )
    best = canonical.snapshot()
    best_loss = math.inf
    stale = 0
    diagnostics = {}
    for step in range(int(stage_cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        output = context.model(**canonical.smplx_kwargs(context.beta))
        vertex_residual = torch.linalg.vector_norm(output.vertices - target_vertices, dim=-1)
        vertex_loss = (
            vertex_residual * context.regions.vertex_weights.unsqueeze(0)
        ).sum() / context.regions.vertex_weights.sum()
        joint_loss = centered_point_loss(output.joints[:, :76], fitted_joints[:, :76])
        evidence_terms = {"body_2d", "hand_2d"}
        if cfg["method"].get("contact", False):
            evidence_terms |= {"contact", "penetration"}
        evidence, raw, weighted, _ = _objective(context, canonical, evidence_terms)
        total = vertex_loss + 10.0 * joint_loss + 0.10 * evidence
        if not torch.isfinite(total):
            canonical.restore(best)
            raise FloatingPointError(f"S5_canonical_refit non-finite at {step}")
        total.backward()
        mask_body_gradient(canonical, UPPER_BODY_SLOTS)
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 5.0)
        optimizer.step()
        current = float(total.detach())
        raw_log = dict(raw)
        raw_log.update({"canonical_vertices": vertex_loss, "canonical_joints": joint_loss})
        weighted_log = dict(weighted)
        weighted_log.update({"canonical_vertices": vertex_loss, "canonical_joints": 10.0 * joint_loss})
        _write_step(log_handle, "S5_canonical_refit", step, total, raw_log, weighted_log, float(grad_norm))
        if current + 1e-8 < best_loss:
            best_loss, best, stale = current, canonical.snapshot(), 0
        else:
            stale += 1
        if step + 1 >= minimum_steps and stale >= 10:
            break
    canonical.restore(best)
    with torch.no_grad():
        output = context.model(**canonical.smplx_kwargs(context.beta))
        residual = torch.linalg.vector_norm(output.vertices - target_vertices, dim=-1)
        hand_ids = torch.cat((context.regions.indices["left_hand"], context.regions.indices["right_hand"]))
        diagnostics = {
            "best_loss": best_loss,
            "hand_residual_mm": float(residual[:, hand_ids].mean() * 1000),
            "upper_residual_mm": float(residual[:, context.regions.upper_vertices].mean() * 1000),
        }
    return canonical, diagnostics


def fit_frame_full(
    observation: FrameObservation,
    model,
    beta,
    camera,
    regions: GeometryRegions,
    cfg: dict,
    output_path: Path,
    log_path: Path,
    device: str,
) -> dict[str, object]:
    import torch

    initial = observation_initial_state(observation, device)
    ablation = cfg["method"]
    use_hypotheses = bool(ablation.get("hypotheses", False))
    use_contact = bool(ablation.get("contact", False))
    proposals = build_contact_proposals(
        observation, float(cfg["contact"].get("min_confidence", 0.70))
    ) if use_contact else []
    context = FitContext(
        observation, model, beta, camera, regions, model.faces_tensor.to(device),
        initial.clone(), {key: float(value) for key, value in cfg["loss"].items()}, proposals, device,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        state = initial.clone()
        s0 = cfg["optimization"]["stages"]["camera_root"]
        optimize_adam(
            context, state, [state.transl, state.global_orient],
            {"body_2d", "root_anchor", "pose_anchor"}, s0["steps"], s0["lr"], (),
            "S0_camera_root", log_handle, cfg["optimization"].get("grad_clip", 5.0),
            cfg["optimization"].get("early_stop_patience", 20),
            s0.get("min_steps", 50),
        )
        s1 = cfg["optimization"]["stages"]["upper_body"]
        optimize_adam(
            context, state, [state.body_pose],
            {"body_2d", "body_teacher", "arm_chain", "anatomy", "pose_anchor"},
            s1["steps"], s1["lr"], UPPER_BODY_SLOTS, "S1_upper_body", log_handle,
            cfg["optimization"].get("grad_clip", 5.0), cfg["optimization"].get("early_stop_patience", 20),
            s1.get("min_steps", 75),
        )
        selected_names = {"left": "h4w_single", "right": "h4w_single"}
        if use_hypotheses:
            degrees = tuple(float(x) for x in cfg["hypotheses"]["wrist_twist_degrees"])
            coarse_keep = int(cfg["hypotheses"].get("coarse_keep", 4))
            left_k1 = _fit_k1(context, _rank_k0(context, state, "left", degrees, coarse_keep, log_handle), "left", cfg, log_handle)
            right_k1 = _fit_k1(context, _rank_k0(context, state, "right", degrees, coarse_keep, log_handle), "right", cfg, log_handle)
            if not left_k1 or not right_k1:
                raise RuntimeError(f"{observation.record.sign}/{observation.record.source_frame_id}: no finite K1 hypothesis")
            pair_results = []
            max_pairs = int(cfg["hypotheses"].get("max_pair_combinations", 4))
            s3 = cfg["optimization"]["stages"]["bimanual_contact"]
            enabled = {
                "body_2d", "hand_2d", "body_teacher", "hand_teacher", "arm_chain",
                "palm_chirality", "anatomy", "pose_anchor",
            }
            if use_contact:
                enabled |= {"contact", "penetration"}
            for left, right in list(product(left_k1, right_k1))[:max_pairs]:
                pair = _combine_pair(left, right)
                optimize_adam(
                    context, pair, [pair.body_pose, pair.left_hand_pose, pair.right_hand_pose], enabled,
                    s3["steps"], s3["lr"], S3_BODY_SLOTS,
                    f"S3_bimanual_contact_{left.name}__{right.name}", log_handle,
                    cfg["optimization"].get("grad_clip", 5.0), cfg["optimization"].get("early_stop_patience", 20),
                    s3.get("min_steps", 80),
                )
                score, components = _score(context, pair, enabled)
                pair_results.append((score, left.name, right.name, pair, components))
            pair_results.sort(key=lambda item: (
                item[0], item[4].get("penetration", 0.0), item[4].get("hand_teacher", 0.0),
                item[4].get("anatomy", 0.0), item[1], item[2],
            ))
            _, left_name, right_name, state, _ = pair_results[0]
            selected_names = {"left": left_name, "right": right_name}
        full_enabled = {
            "body_2d", "hand_2d", "body_teacher", "hand_teacher", "arm_chain",
            "anatomy", "pose_anchor",
        }
        if use_hypotheses:
            full_enabled.add("palm_chirality")
        if use_contact:
            full_enabled |= {"contact", "penetration"}
        lbfgs = cfg["optimization"]["stages"]["lbfgs"]
        lbfgs_accepted = False
        if bool(lbfgs.get("enabled", True)):
            lbfgs_accepted, _ = optimize_lbfgs(
                context, state, [state.body_pose, state.left_hand_pose, state.right_hand_pose],
                full_enabled, lbfgs["steps"], lbfgs["lr"], S3_BODY_SLOTS, log_handle,
            )
        canonical, canonical_diagnostics = _canonical_refit(context, state, cfg, log_handle)
        with torch.no_grad():
            final_output = model(**canonical.smplx_kwargs(beta))
            final_total, final_raw, _, _ = _objective(context, canonical, full_enabled)
        if not torch.isfinite(final_output.vertices).all() or not torch.isfinite(final_total):
            raise FloatingPointError("final canonical state is non-finite")
    arrays = {
        "frame_ids": np.asarray([observation.record.source_frame_id], dtype=np.int64),
        "mesh_parametric": final_output.vertices.detach().cpu().numpy().astype(np.float32),
        "betas": beta.detach().cpu().numpy().reshape(1, 10).astype(np.float32),
        "global_orient": canonical.global_orient.detach().cpu().numpy().reshape(1, 3).astype(np.float32),
        "body_pose": canonical.body_pose.detach().cpu().numpy().reshape(1, 63).astype(np.float32),
        "left_hand_pose": canonical.left_hand_pose.detach().cpu().numpy().reshape(1, 45).astype(np.float32),
        "right_hand_pose": canonical.right_hand_pose.detach().cpu().numpy().reshape(1, 45).astype(np.float32),
        "jaw_pose": canonical.jaw_pose.detach().cpu().numpy().reshape(1, 3).astype(np.float32),
        "leye_pose": np.zeros((1, 3), dtype=np.float32), "reye_pose": np.zeros((1, 3), dtype=np.float32),
        "expression": canonical.expression.detach().cpu().numpy().reshape(1, 10).astype(np.float32),
        "transl": canonical.transl.detach().cpu().numpy().reshape(1, 3).astype(np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    report = {
        "schema_version": "signpccx.full-staged-frame.v1",
        "sign": observation.record.sign, "frame_id": observation.record.source_frame_id,
        "source": str(observation.cache_path.resolve()), "source_sha256": sha256_file(observation.cache_path),
        "stages": ["S0_camera_root", "S1_upper_body", "S2_hand_hypothesis" if use_hypotheses else "S2_disabled_ablation",
                   "S3_bimanual_contact" if use_hypotheses else "S3_disabled_ablation",
                   "S4_lbfgs_refine", "S5_canonical_refit"],
        "selected_hypotheses": selected_names, "contact_proposals": [proposal.__dict__ for proposal in proposals],
        "lbfgs_accepted": lbfgs_accepted, "canonical": canonical_diagnostics,
        "final_objective": float(final_total.detach().cpu()),
        "final_raw_losses": {
            key: float(value.detach().cpu()) for key, value in final_raw.items()
            if torch.is_tensor(value) and value.numel() == 1
        },
        "objective_uses_ground_truth": False, "objective_uses_temporal_pose": False,
        "sha256": sha256_file(output_path), "log_sha256": sha256_file(log_path),
    }
    atomic_write_json(output_path.with_suffix(".json"), report)
    return report


def _load_identity(path: Path, device: str):
    import torch

    with np.load(path, allow_pickle=False) as archive:
        beta = torch.as_tensor(archive["beta"], dtype=torch.float32, device=device).reshape(1, 10)
        focal = float(archive["focal_magnitude"])
        principal = tuple(np.asarray(archive["principal"], dtype=np.float32).tolist())
        image_wh = tuple(np.asarray(archive["image_wh"], dtype=np.int64).tolist())
        signs = tuple(np.asarray(archive["focal_signs"], dtype=np.float32).tolist())
    camera = SharedSignedCamera(focal, principal, image_wh, signs).to(device)
    for parameter in camera.parameters():
        parameter.requires_grad_(False)
    return beta, camera


def _all_records(manifest_root: Path, signs: set[str] | None = None) -> list[FrameRecord]:
    paths = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        unknown = signs - {path.stem for path in paths}
        if unknown:
            raise ValueError(f"unknown signs: {sorted(unknown)}")
        paths = [path for path in paths if path.stem in signs]
    return [record for path in paths for record in read_jsonl(path)]


def calibrate_signer_full(
    manifest_root: Path,
    cache_root: Path,
    initializer_root: Path,
    model_root: Path,
    output_path: Path,
    cfg: dict,
    device: str,
) -> dict[str, object]:
    """Run the blueprint's four-phase alternating shared beta/camera fit."""
    import torch
    import smplx
    from signpccx.optimization.losses import affine_homogeneous, arm_chain_loss, centered_point_loss, keypoint_loss, safe_project

    validate_body_slots()
    records = _all_records(manifest_root)
    observations = [load_frame_observation(record, cache_root, initializer_root) for record in records]
    features = []
    for observation in observations:
        a = observation.arrays
        anchors = a["init_anchor_cam"]
        shoulder_width = np.linalg.norm(anchors[2] - anchors[3])
        hand_distance = np.linalg.norm(anchors[8] - anchors[19])
        feature = np.concatenate((
            [shoulder_width, hand_distance, a["person_bbox_xywh"][2] / max(a["person_bbox_xywh"][3], 1e-6)],
            (anchors[[4, 5, 6, 7], :2] - anchors[1, :2]).reshape(-1),
        ))
        features.append(feature)
    selected_indices = farthest_point_indices(np.stack(features), int(cfg["identity"]["calibration_frames"]))
    selected = [observations[index] for index in selected_indices]
    beta0 = huber_location(np.stack([item.arrays["smplx_beta"] for item in selected]))
    cameras = np.stack([_full_intrinsics(item) for item in selected])
    f0 = float(np.median(np.abs(cameras[:, [0, 1], [0, 1]])))
    principal0 = tuple(np.median(cameras[:, :2, 2], axis=0).tolist())
    image_wh = selected[0].image_wh
    camera = SharedSignedCamera(
        f0, principal0, image_wh, (-1.0, 1.0),
        tuple(cfg["identity"].get("focal_bounds", [0.5, 2.0])),
        float(cfg["identity"].get("max_principal_shift_fraction", 0.05)),
    ).to(device)
    beta = torch.nn.Parameter(_tensor(beta0, device).reshape(1, 10))
    count = len(selected)
    root = torch.nn.Parameter(torch.stack([observation_initial_state(item, device).global_orient[0] for item in selected]))
    body = torch.nn.Parameter(torch.stack([observation_initial_state(item, device).body_pose[0] for item in selected]))
    transl = torch.nn.Parameter(torch.stack([observation_initial_state(item, device).transl[0] for item in selected]))
    fixed_left = torch.stack([observation_initial_state(item, device).left_hand_pose[0] for item in selected])
    fixed_right = torch.stack([observation_initial_state(item, device).right_hand_pose[0] for item in selected])
    jaw = torch.stack([observation_initial_state(item, device).jaw_pose[0] for item in selected])
    expression = torch.stack([observation_initial_state(item, device).expression[0] for item in selected])
    root0, body0, transl0 = root.detach().clone(), body.detach().clone(), transl.detach().clone()
    model = smplx.create(str(model_root), model_type="smplx", gender="neutral", num_betas=10,
                         use_pca=False, use_face_contour=True, batch_size=count).to(device)
    model.eval()
    target_joints = torch.stack([_tensor(item.arrays["smplx_joints_parametric"], device) for item in selected])
    observed_uv = torch.stack([_tensor(item.arrays["anchor_uv_observed"], device) for item in selected])
    confidence = torch.stack([_tensor(item.arrays["anchor_uv_confidence"] * item.arrays["anchor_valid"], device) for item in selected])
    transforms = torch.stack([affine_homogeneous(_tensor(item.arrays["image_to_crop"], device))[0] for item in selected])
    beta_anchor = beta.detach().clone()
    log_path = output_path.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def forward():
        zero_eye = torch.zeros((count, 3), dtype=torch.float32, device=device)
        return model(
            global_orient=root, body_pose=body.reshape(count, 63), left_hand_pose=fixed_left.reshape(count, 45),
            right_hand_pose=fixed_right.reshape(count, 45), jaw_pose=jaw, leye_pose=zero_eye, reye_pose=zero_eye,
            expression=expression, betas=beta.expand(count, -1), transl=transl, return_verts=True,
        )

    def objective():
        output = forward()
        anchors = _anchor_joints(output.joints)
        intrinsics = transforms @ camera.matrix().unsqueeze(0)
        uv = safe_project(anchors, intrinsics)
        body2d = keypoint_loss(uv[:, :8], observed_uv[:, :8], confidence[:, :8], selected[0].crop_hw)
        teacher = centered_point_loss(output.joints[:, list(BODY_JOINT_IDS)], target_joints[:, list(BODY_JOINT_IDS)])
        chain = 0.5 * sum(arm_chain_loss(output.joints, target_joints, CHAIN_IDS[side], confidence[:, :8].mean(1)) for side in ("left", "right"))
        beta_prior = (beta - beta_anchor).square().mean()
        focal_prior = (camera.log_f - math.log(f0)).square()
        principal_prior = camera.delta_c.square().mean()
        pose_anchor = (root - root0).square().mean() + (body - body0).square().mean() + 0.01 * (transl - transl0).square().mean()
        terms = {"body_2d": body2d, "body_teacher": teacher, "bone_ratio": chain,
                 "beta_prior": beta_prior, "focal_prior": focal_prior,
                 "principal_prior": principal_prior, "pose_anchor": pose_anchor}
        weights = cfg["identity"].get("weights", {})
        total = sum(float(weights.get(name, 1.0)) * value for name, value in terms.items())
        return total, terms

    schedule = tuple(int(value) for value in cfg["identity"]["alternating_steps"])
    phases = (
        ("camera_pose", schedule[0], [
            {"params": [root, body, transl], "lr": 1e-2},
            {"params": [camera.log_f, camera.delta_c], "lr": 5e-4},
        ]),
        ("beta_pose", schedule[1], [
            {"params": [beta, body, transl], "lr": 3e-3},
        ]),
        ("camera_refine", schedule[2], [
            {"params": [camera.log_f, camera.delta_c], "lr": 5e-4},
        ]),
        ("joint_trust_region", schedule[3], [
            {"params": [beta, root, body, transl], "lr": 1e-3},
            {"params": [camera.log_f, camera.delta_c], "lr": 1e-4},
        ]),
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        for phase, steps, parameter_groups in phases:
            optimizer = torch.optim.Adam(parameter_groups)
            parameters = [parameter for group in parameter_groups for parameter in group["params"]]
            best = math.inf
            best_state = {
                "beta": beta.detach().clone(), "root": root.detach().clone(),
                "body": body.detach().clone(), "transl": transl.detach().clone(),
                "camera": {key: value.detach().clone() for key, value in camera.state_dict().items()},
            }
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                total, terms = objective()
                if not torch.isfinite(total):
                    raise FloatingPointError(f"calibration {phase}/{step} non-finite")
                total.backward()
                if body.grad is not None:
                    mask = torch.zeros_like(body.grad)
                    mask[:, list(UPPER_BODY_SLOTS)] = 1
                    body.grad.mul_(mask)
                grad = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                current = float(total.detach())
                if current < best:
                    best = current
                    best_state = {
                        "beta": beta.detach().clone(), "root": root.detach().clone(),
                        "body": body.detach().clone(), "transl": transl.detach().clone(),
                        "camera": {key: value.detach().clone() for key, value in camera.state_dict().items()},
                    }
                _write_step(handle, f"calibration_{phase}", step, total, terms,
                            {name: terms[name] * float(cfg["identity"].get("weights", {}).get(name, 1.0)) for name in terms},
                            float(grad))
            with torch.no_grad():
                beta.copy_(best_state["beta"])
                root.copy_(best_state["root"])
                body.copy_(best_state["body"])
                transl.copy_(best_state["transl"])
            camera.load_state_dict(best_state["camera"])
            handle.write(json.dumps({
                "stage": f"calibration_{phase}", "event": "restore_best", "best": best,
            }, sort_keys=True) + "\n")
            handle.flush()
    matrix = camera.matrix().detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, beta=beta.detach().cpu().numpy().reshape(10).astype(np.float32),
        robust_beta=beta0, focal_magnitude=np.asarray(abs(matrix[0, 0]), dtype=np.float32),
        principal=matrix[:2, 2].astype(np.float32), focal_signs=np.asarray([-1.0, 1.0], dtype=np.float32),
        image_wh=np.asarray(image_wh, dtype=np.int64), selected_indices=selected_indices,
        selected_frame_ids=np.asarray([item.record.source_frame_id for item in selected], dtype=np.int64),
    )
    report = {
        "schema_version": "signpccx.full-calibration.v1", "frames": count,
        "selection": "farthest_point_pose_diversity", "alternating_steps": list(schedule),
        "beta": beta.detach().cpu().numpy().reshape(10).tolist(),
        "camera_matrix": matrix.tolist(), "log_sha256": sha256_file(log_path),
        "objective_uses_ground_truth": False, "objective_uses_temporal_pose": False,
        "selected": [{"sign": item.record.sign, "frame_id": item.record.source_frame_id} for item in selected],
        "sha256": sha256_file(output_path),
    }
    atomic_write_json(output_path.with_suffix(".json"), report)
    return report


def fit_signs_full(
    manifest_root: Path,
    cache_root: Path,
    initializer_root: Path,
    identity_path: Path,
    model_root: Path,
    mano_ids_path: Path,
    output_root: Path,
    cfg: dict,
    device: str,
    signs: set[str] | None = None,
    limit: int | None = None,
    frame_ids: set[int] | None = None,
) -> dict[str, object]:
    import smplx
    import torch

    validate_body_slots()
    torch.manual_seed(int(cfg["experiment"].get("seed", 20260830)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["experiment"].get("seed", 20260830)))
    if cfg["experiment"].get("deterministic", True):
        torch.use_deterministic_algorithms(True, warn_only=True)
    beta_shared, camera_shared = _load_identity(identity_path, device)
    model = smplx.create(str(model_root), model_type="smplx", gender="neutral", num_betas=10,
                         use_pca=False, use_face_contour=True).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    regions = build_geometry_regions(
        model, mano_ids_path, device, int(cfg["contact"].get("vertices_per_tip_region", 24))
    )
    manifest_paths = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        manifest_paths = [path for path in manifest_paths if path.stem in signs]
    reports = []
    processed = 0
    for manifest_path in manifest_paths:
        records = read_jsonl(manifest_path)
        sign_reports = []
        for record in records:
            if frame_ids is not None and record.source_frame_id not in frame_ids:
                continue
            if limit is not None and processed >= limit:
                break
            frame_root = output_root / "frames" / record.sign
            output_path = frame_root / f"{record.source_frame_id:06d}.npz"
            sidecar = output_path.with_suffix(".json")
            log_path = output_root / "logs" / record.sign / f"{record.source_frame_id:06d}.jsonl"
            if output_path.is_file() and sidecar.is_file():
                cached = json.loads(sidecar.read_text(encoding="utf-8"))
                if cached.get("sha256") != sha256_file(output_path) or cached.get("log_sha256") != sha256_file(log_path):
                    raise RuntimeError(f"invalid resume artifact: {output_path}")
                sign_reports.append(cached)
                processed += 1
                continue
            observation = load_frame_observation(record, cache_root, initializer_root)
            beta = beta_shared if cfg["method"].get("shared_beta", False) else _tensor(observation.arrays["smplx_beta"], device).reshape(1, 10)
            camera = camera_shared if cfg["method"].get("shared_camera", False) else None
            report = fit_frame_full(observation, model, beta, camera, regions, cfg, output_path, log_path, device)
            sign_reports.append(report)
            processed += 1
        if frame_ids is None and len(sign_reports) == len(records):
            arrays_per_frame = []
            for record in records:
                with np.load(output_root / "frames" / record.sign / f"{record.source_frame_id:06d}.npz", allow_pickle=False) as archive:
                    arrays_per_frame.append({key: np.asarray(archive[key]).copy() for key in archive.files})
            keys = arrays_per_frame[0].keys()
            combined = {key: np.concatenate([item[key] for item in arrays_per_frame], axis=0) for key in keys}
            destination = output_root / "clips" / manifest_path.stem / "mesh_parametric_final.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **combined)
            sign_summary = {
                "schema_version": "signpccx.full-staged-sign.v1", "sign": manifest_path.stem,
                "frames": len(records), "sha256": sha256_file(destination),
                "frame_sidecars": [str((output_root / "frames" / record.sign / f"{record.source_frame_id:06d}.json").resolve()) for record in records],
            }
            atomic_write_json(destination.with_suffix(".json"), sign_summary)
            reports.append(sign_summary)
        if limit is not None and processed >= limit:
            break
    summary = {
        "schema_version": "signpccx.full-staged-summary.v1", "processed_frames": processed,
        "completed_signs": len(reports), "method": cfg["method"], "items": reports,
    }
    atomic_write_json(output_root / "fit_summary.json", summary)
    return summary
