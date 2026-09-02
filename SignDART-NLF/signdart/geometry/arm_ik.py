from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ray_sphere import enumerate_arm_branches, enumerate_three_link_branches, project
from .rotations import (
    geodesic_angle_deg,
    global_matrices,
    local_matrices,
    matrices_to_body_pose,
    rotation_between,
)


BOUNDARY_X180 = np.diag([1.0, -1.0, -1.0])
ARM_IDS = {
    "left": {"collar": 13, "shoulder": 16, "elbow": 18, "wrist": 20},
    "right": {"collar": 14, "shoulder": 17, "elbow": 19, "wrist": 21},
}


@dataclass(frozen=True)
class ArmCandidate:
    name: str
    side: str
    body_pose: np.ndarray
    shoulder_target: np.ndarray
    elbow_target: np.ndarray
    wrist_target: np.ndarray
    incumbent_equivalent: bool
    global_wrist_error_deg: float


def internal_intrinsics(evaluator_K: np.ndarray) -> np.ndarray:
    """Map internal positive-z SMPL-X joints to the evaluator image pixels."""
    return np.asarray(evaluator_K, dtype=np.float64).reshape(3, 3) @ BOUNDARY_X180


def solve_arm_body_pose(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    parents: np.ndarray,
    joints_internal: np.ndarray,
    side: str,
    elbow_target: np.ndarray,
    wrist_target: np.ndarray,
) -> tuple[np.ndarray, float]:
    ids = ARM_IDS[side]
    shoulder_id, elbow_id, wrist_id = (
        ids["shoulder"], ids["elbow"], ids["wrist"]
    )
    local = local_matrices(global_orient, body_pose)
    incumbent_global = global_matrices(local, parents)
    candidate = local.copy()

    shoulder = np.asarray(joints_internal[shoulder_id], dtype=np.float64)
    elbow0 = np.asarray(joints_internal[elbow_id], dtype=np.float64)
    wrist0 = np.asarray(joints_internal[wrist_id], dtype=np.float64)
    elbow_target = np.asarray(elbow_target, dtype=np.float64)
    wrist_target = np.asarray(wrist_target, dtype=np.float64)

    shoulder_swing = rotation_between(elbow0 - shoulder, elbow_target - shoulder)
    shoulder_global = shoulder_swing @ incumbent_global[shoulder_id]
    shoulder_parent = int(parents[shoulder_id])
    candidate[shoulder_id] = incumbent_global[shoulder_parent].T @ shoulder_global

    elbow_pre = shoulder_global @ local[elbow_id]
    forearm_local = incumbent_global[elbow_id].T @ (wrist0 - elbow0)
    forearm_pre = elbow_pre @ forearm_local
    elbow_swing = rotation_between(forearm_pre, wrist_target - elbow_target)
    elbow_global = elbow_swing @ elbow_pre
    candidate[elbow_id] = shoulder_global.T @ elbow_global

    candidate[wrist_id] = elbow_global.T @ incumbent_global[wrist_id]
    candidate_global = global_matrices(candidate, parents)
    wrist_error = geodesic_angle_deg(
        candidate_global[wrist_id], incumbent_global[wrist_id]
    )
    return matrices_to_body_pose(candidate), wrist_error


def solve_three_link_body_pose(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    parents: np.ndarray,
    joints_internal: np.ndarray,
    side: str,
    shoulder_target: np.ndarray,
    elbow_target: np.ndarray,
    wrist_target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Realize a three-link target while retaining H1's global wrist frame."""
    ids = ARM_IDS[side]
    collar_id, shoulder_id, elbow_id, wrist_id = (
        ids["collar"], ids["shoulder"], ids["elbow"], ids["wrist"]
    )
    local = local_matrices(global_orient, body_pose)
    incumbent_global = global_matrices(local, parents)
    candidate = local.copy()

    collar = np.asarray(joints_internal[collar_id], dtype=np.float64)
    shoulder0 = np.asarray(joints_internal[shoulder_id], dtype=np.float64)
    elbow0 = np.asarray(joints_internal[elbow_id], dtype=np.float64)
    wrist0 = np.asarray(joints_internal[wrist_id], dtype=np.float64)
    shoulder_target = np.asarray(shoulder_target, dtype=np.float64)
    elbow_target = np.asarray(elbow_target, dtype=np.float64)
    wrist_target = np.asarray(wrist_target, dtype=np.float64)

    collar_swing = rotation_between(
        shoulder0 - collar, shoulder_target - collar
    )
    collar_global = collar_swing @ incumbent_global[collar_id]
    collar_parent = int(parents[collar_id])
    candidate[collar_id] = incumbent_global[collar_parent].T @ collar_global

    shoulder_pre = collar_global @ local[shoulder_id]
    upper_local = incumbent_global[shoulder_id].T @ (elbow0 - shoulder0)
    upper_pre = shoulder_pre @ upper_local
    shoulder_swing = rotation_between(
        upper_pre, elbow_target - shoulder_target
    )
    shoulder_global = shoulder_swing @ shoulder_pre
    candidate[shoulder_id] = collar_global.T @ shoulder_global

    elbow_pre = shoulder_global @ local[elbow_id]
    forearm_local = incumbent_global[elbow_id].T @ (wrist0 - elbow0)
    forearm_pre = elbow_pre @ forearm_local
    elbow_swing = rotation_between(forearm_pre, wrist_target - elbow_target)
    elbow_global = elbow_swing @ elbow_pre
    candidate[elbow_id] = shoulder_global.T @ elbow_global

    candidate[wrist_id] = elbow_global.T @ incumbent_global[wrist_id]
    candidate_global = global_matrices(candidate, parents)
    wrist_error = geodesic_angle_deg(
        candidate_global[wrist_id], incumbent_global[wrist_id]
    )
    return matrices_to_body_pose(candidate), wrist_error


def enumerate_body_pose_candidates(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    parents: np.ndarray,
    joints_internal: np.ndarray,
    evaluator_K: np.ndarray,
    side: str,
) -> list[ArmCandidate]:
    ids = ARM_IDS[side]
    shoulder = np.asarray(joints_internal[ids["shoulder"]], dtype=np.float64)
    elbow = np.asarray(joints_internal[ids["elbow"]], dtype=np.float64)
    wrist = np.asarray(joints_internal[ids["wrist"]], dtype=np.float64)
    K = internal_intrinsics(evaluator_K)
    elbow_uv = project(K, elbow)
    wrist_uv = project(K, wrist)
    branches = enumerate_arm_branches(
        shoulder,
        elbow_uv,
        wrist_uv,
        float(np.linalg.norm(elbow - shoulder)),
        float(np.linalg.norm(wrist - elbow)),
        K,
    )
    output = [
        ArmCandidate(
            name="c0",
            side=side,
            body_pose=np.asarray(body_pose, dtype=np.float32).reshape(63).copy(),
            shoulder_target=shoulder.copy(),
            elbow_target=elbow.copy(),
            wrist_target=wrist.copy(),
            incumbent_equivalent=True,
            global_wrist_error_deg=0.0,
        )
    ]
    for branch in branches:
        elbow_target = np.asarray(branch["elbow"], dtype=np.float64)
        wrist_target = np.asarray(branch["wrist"], dtype=np.float64)
        equivalent = bool(
            np.linalg.norm(elbow_target - elbow) <= 1e-5
            and np.linalg.norm(wrist_target - wrist) <= 1e-5
        )
        if equivalent:
            continue
        candidate_pose, wrist_error = solve_arm_body_pose(
            global_orient,
            body_pose,
            parents,
            joints_internal,
            side,
            elbow_target,
            wrist_target,
        )
        output.append(
            ArmCandidate(
                name=str(branch["name"]),
                side=side,
                body_pose=candidate_pose,
                shoulder_target=shoulder.copy(),
                elbow_target=elbow_target,
                wrist_target=wrist_target,
                incumbent_equivalent=False,
                global_wrist_error_deg=wrist_error,
            )
        )
    return output


def enumerate_three_link_body_pose_candidates(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    parents: np.ndarray,
    joints_internal: np.ndarray,
    evaluator_K: np.ndarray,
    side: str,
) -> list[ArmCandidate]:
    ids = ARM_IDS[side]
    collar = np.asarray(joints_internal[ids["collar"]], dtype=np.float64)
    shoulder = np.asarray(joints_internal[ids["shoulder"]], dtype=np.float64)
    elbow = np.asarray(joints_internal[ids["elbow"]], dtype=np.float64)
    wrist = np.asarray(joints_internal[ids["wrist"]], dtype=np.float64)
    K = internal_intrinsics(evaluator_K)
    branches = enumerate_three_link_branches(
        collar,
        project(K, shoulder),
        project(K, elbow),
        project(K, wrist),
        float(np.linalg.norm(shoulder - collar)),
        float(np.linalg.norm(elbow - shoulder)),
        float(np.linalg.norm(wrist - elbow)),
        K,
    )
    output = [
        ArmCandidate(
            name="c0",
            side=side,
            body_pose=np.asarray(body_pose, dtype=np.float32).reshape(63).copy(),
            shoulder_target=shoulder.copy(),
            elbow_target=elbow.copy(),
            wrist_target=wrist.copy(),
            incumbent_equivalent=True,
            global_wrist_error_deg=0.0,
        )
    ]
    for branch in branches:
        shoulder_target = np.asarray(branch["shoulder"], dtype=np.float64)
        elbow_target = np.asarray(branch["elbow"], dtype=np.float64)
        wrist_target = np.asarray(branch["wrist"], dtype=np.float64)
        equivalent = bool(
            np.linalg.norm(shoulder_target - shoulder) <= 1e-5
            and np.linalg.norm(elbow_target - elbow) <= 1e-5
            and np.linalg.norm(wrist_target - wrist) <= 1e-5
        )
        if equivalent:
            continue
        candidate_pose, wrist_error = solve_three_link_body_pose(
            global_orient,
            body_pose,
            parents,
            joints_internal,
            side,
            shoulder_target,
            elbow_target,
            wrist_target,
        )
        output.append(
            ArmCandidate(
                name=str(branch["name"]),
                side=side,
                body_pose=candidate_pose,
                shoulder_target=shoulder_target,
                elbow_target=elbow_target,
                wrist_target=wrist_target,
                incumbent_equivalent=False,
                global_wrist_error_deg=wrist_error,
            )
        )
    return output
