from __future__ import annotations

import numpy as np


CHAIN_IDS = {
    "left": (13, 16, 18, 20),
    "right": (14, 17, 19, 21),
}
CHAIN_SLOTS = {
    "left": (12, 15, 17, 19),
    "right": (13, 16, 18, 20),
}

# Fixed 95% chi-square quantiles for one to three changed arm bones.  These are
# statistical constants, not values estimated from the evaluation benchmark.
CHI_SQUARE_95 = {
    1: 3.841458820694124,
    2: 5.991464547107979,
    3: 7.814727903251179,
}
NORMAL_95 = 1.959963984540054


def normalized_bones(joints: np.ndarray, ids: tuple[int, ...]) -> np.ndarray:
    points = np.asarray(joints, dtype=np.float64)[list(ids)]
    bones = points[1:] - points[:-1]
    lengths = np.linalg.norm(bones, axis=-1, keepdims=True)
    return bones / np.maximum(lengths, 1e-12)


def uncertainty_weights(
    parametric_mm: np.ndarray,
    nonparametric_mm: np.ndarray,
    uncertainty_mm: np.ndarray,
    ids: tuple[int, ...],
) -> np.ndarray:
    ids_array = np.asarray(ids, dtype=np.int64)
    fit = np.linalg.norm(
        np.asarray(parametric_mm)[ids_array] - np.asarray(nonparametric_mm)[ids_array],
        axis=-1,
    )
    uncertainty = np.asarray(uncertainty_mm, dtype=np.float64)[ids_array]
    effective = 1e-6 + 0.5 * (uncertainty[:-1] + uncertainty[1:])
    effective += 0.5 * (fit[:-1] + fit[1:])
    weights = np.square(1.0 / effective)
    return weights / weights.sum()


def branch_scores(
    candidate_joints: np.ndarray,
    nlf_parametric_mm: np.ndarray,
    nlf_nonparametric_mm: np.ndarray,
    nlf_uncertainty_mm: np.ndarray,
    side: str,
) -> np.ndarray:
    ids = CHAIN_IDS[side]
    evidence = normalized_bones(nlf_nonparametric_mm, ids)
    weights = uncertainty_weights(
        nlf_parametric_mm, nlf_nonparametric_mm, nlf_uncertainty_mm, ids
    )
    candidates = np.stack([normalized_bones(joints, ids) for joints in candidate_joints])
    cosine = np.clip(np.einsum("nbd,bd->nb", candidates, evidence), -1.0, 1.0)
    return np.sum(weights[None] * (1.0 - cosine), axis=1)


def angular_uncertainty(
    joints_mm: np.ndarray,
    uncertainty_mm: np.ndarray,
    ids: tuple[int, ...],
) -> np.ndarray:
    """Propagate endpoint position uncertainty to a first-order bone angle."""
    ids_array = np.asarray(ids, dtype=np.int64)
    points = np.asarray(joints_mm, dtype=np.float64)[ids_array]
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    uncertainty = np.asarray(uncertainty_mm, dtype=np.float64)[ids_array]
    endpoint_uncertainty = np.sqrt(
        np.square(uncertainty[:-1]) + np.square(uncertainty[1:])
    )
    sigma = np.arctan2(endpoint_uncertainty, np.maximum(lengths, 1e-12))
    return np.clip(sigma, 1e-6, np.pi / 2.0)


def angular_residuals(
    candidate_joints: np.ndarray,
    evidence_joints_mm: np.ndarray,
    ids: tuple[int, ...],
) -> np.ndarray:
    candidates = np.stack([
        normalized_bones(joints, ids) for joints in candidate_joints
    ])
    evidence = normalized_bones(evidence_joints_mm, ids)
    cosine = np.clip(np.einsum("nbd,bd->nb", candidates, evidence), -1.0, 1.0)
    return np.arccos(cosine)


def conservative_consensus_decision(
    candidate_joints: np.ndarray,
    parametric_mm: np.ndarray,
    nonparametric_mm: np.ndarray,
    uncertainty_mm: np.ndarray,
    side: str,
    incumbent_index: int = 0,
) -> tuple[int, dict]:
    """Select a branch without fitted parameters, otherwise retain incumbent.

    Parametric and non-parametric NLF predictions must independently prefer the
    same alternative.  The alternative must dominate the incumbent on every
    changed bone, agree with both predictions within their propagated 95%
    uncertainty, and exceed a fixed 95% chi-square likelihood-ratio threshold.
    """
    ids = CHAIN_IDS[side]
    candidates = np.asarray(candidate_joints, dtype=np.float64)
    candidate_dirs = np.stack([normalized_bones(joints, ids) for joints in candidates])
    param_residual = angular_residuals(candidates, parametric_mm, ids)
    nonparam_residual = angular_residuals(candidates, nonparametric_mm, ids)
    param_sigma = angular_uncertainty(parametric_mm, uncertainty_mm, ids)
    nonparam_sigma = angular_uncertainty(nonparametric_mm, uncertainty_mm, ids)

    param_cost = np.sum(np.square(param_residual / param_sigma[None]), axis=1)
    nonparam_cost = np.sum(
        np.square(nonparam_residual / nonparam_sigma[None]), axis=1
    )
    param_best = int(np.argmin(param_cost))
    nonparam_best = int(np.argmin(nonparam_cost))
    diagnostics = {
        "parametric_best_index": param_best,
        "nonparametric_best_index": nonparam_best,
        "parametric_cost_incumbent": float(param_cost[incumbent_index]),
        "nonparametric_cost_incumbent": float(nonparam_cost[incumbent_index]),
        "reason": "retain_incumbent",
    }
    if param_best != nonparam_best:
        diagnostics["reason"] = "estimator_branch_disagreement"
        return incumbent_index, diagnostics
    best = param_best
    if best == incumbent_index:
        diagnostics["reason"] = "incumbent_preferred"
        return incumbent_index, diagnostics

    branch_change = np.arccos(np.clip(
        np.sum(candidate_dirs[best] * candidate_dirs[incumbent_index], axis=-1),
        -1.0,
        1.0,
    ))
    changed = branch_change > 1e-4
    degrees_of_freedom = int(changed.sum())
    if degrees_of_freedom == 0:
        diagnostics["reason"] = "numerically_incumbent_equivalent"
        return incumbent_index, diagnostics

    param_dominates = np.all(
        param_residual[best, changed] < param_residual[incumbent_index, changed]
    )
    nonparam_dominates = np.all(
        nonparam_residual[best, changed]
        < nonparam_residual[incumbent_index, changed]
    )
    if not (param_dominates and nonparam_dominates):
        diagnostics["reason"] = "not_bonewise_dominant"
        return incumbent_index, diagnostics

    evidence_cosine = np.clip(np.sum(
        normalized_bones(parametric_mm, ids)
        * normalized_bones(nonparametric_mm, ids),
        axis=-1,
    ), -1.0, 1.0)
    evidence_disagreement = np.arccos(evidence_cosine)
    agreement_limit = NORMAL_95 * np.sqrt(
        np.square(param_sigma) + np.square(nonparam_sigma)
    )
    if not np.all(evidence_disagreement[changed] <= agreement_limit[changed]):
        diagnostics["reason"] = "estimator_uncertainty_disagreement"
        return incumbent_index, diagnostics

    if not (
        np.all(param_residual[best, changed] <= NORMAL_95 * param_sigma[changed])
        and np.all(
            nonparam_residual[best, changed] <= NORMAL_95 * nonparam_sigma[changed]
        )
    ):
        diagnostics["reason"] = "candidate_outside_uncertainty_interval"
        return incumbent_index, diagnostics

    param_gain = float(np.sum(
        np.square(param_residual[incumbent_index, changed] / param_sigma[changed])
        - np.square(param_residual[best, changed] / param_sigma[changed])
    ))
    nonparam_gain = float(np.sum(
        np.square(
            nonparam_residual[incumbent_index, changed] / nonparam_sigma[changed]
        )
        - np.square(nonparam_residual[best, changed] / nonparam_sigma[changed])
    ))
    threshold = CHI_SQUARE_95[degrees_of_freedom]
    diagnostics.update({
        "degrees_of_freedom": degrees_of_freedom,
        "likelihood_gain_threshold": threshold,
        "parametric_likelihood_gain": param_gain,
        "nonparametric_likelihood_gain": nonparam_gain,
        "parametric_cost_selected": float(param_cost[best]),
        "nonparametric_cost_selected": float(nonparam_cost[best]),
    })
    if param_gain <= threshold or nonparam_gain <= threshold:
        diagnostics["reason"] = "insufficient_likelihood_gain"
        return incumbent_index, diagnostics

    diagnostics["reason"] = "consensus_branch_accepted"
    return best, diagnostics


def compose_selected_pose(
    base_body_pose: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> np.ndarray:
    output = np.asarray(base_body_pose, dtype=np.float32).reshape(21, 3).copy()
    left = np.asarray(left_pose, dtype=np.float32).reshape(21, 3)
    right = np.asarray(right_pose, dtype=np.float32).reshape(21, 3)
    output[list(CHAIN_SLOTS["left"])] = left[list(CHAIN_SLOTS["left"])]
    output[list(CHAIN_SLOTS["right"])] = right[list(CHAIN_SLOTS["right"])]
    return output.reshape(63)
