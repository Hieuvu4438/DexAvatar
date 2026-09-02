from __future__ import annotations

import numpy as np

from .selector import CHAIN_IDS, normalized_bones, uncertainty_weights


FEATURE_NAMES = (
    "nlf_score", "nlf_score_gain_vs_c0", "distal_risk_mm", "side_right",
    "uncertainty_mean_mm", "param_nonparam_fit_mean_mm", "box_score",
    "box_area_fraction",
    *(f"candidate_nlf_cos_{i}" for i in range(3)),
    *(f"h1_nlf_cos_{i}" for i in range(3)),
    *(f"candidate_h1_cos_{i}" for i in range(3)),
    *(f"relative_depth_{i}" for i in range(3)),
    *(f"candidate_direction_{i}_{axis}" for i in range(3) for axis in "xyz"),
    *(f"nlf_direction_{i}_{axis}" for i in range(3) for axis in "xyz"),
    *(f"bone_reliability_{i}" for i in range(3)),
    *(f"param_nonparam_cos_{i}" for i in range(3)),
)


def candidate_features(
    candidate_joints: np.ndarray,
    candidate_metrics: np.ndarray,
    nlf_parametric_mm: np.ndarray,
    nlf_nonparametric_mm: np.ndarray,
    nlf_uncertainty_mm: np.ndarray,
    side: str,
    box: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    ids = CHAIN_IDS[side]
    candidates = np.asarray(candidate_joints, dtype=np.float64)
    candidate_dirs = np.stack([normalized_bones(joints, ids) for joints in candidates])
    h1_dirs = candidate_dirs[0]
    nlf_dirs = normalized_bones(nlf_nonparametric_mm, ids)
    param_dirs = normalized_bones(nlf_parametric_mm, ids)
    weights = uncertainty_weights(
        nlf_parametric_mm, nlf_nonparametric_mm, nlf_uncertainty_mm, ids
    )
    candidate_nlf = np.clip(
        np.einsum("nbd,bd->nb", candidate_dirs, nlf_dirs), -1.0, 1.0
    )
    h1_nlf = np.clip(np.sum(h1_dirs * nlf_dirs, axis=-1), -1.0, 1.0)
    candidate_h1 = np.clip(
        np.einsum("nbd,bd->nb", candidate_dirs, h1_dirs), -1.0, 1.0
    )
    param_nonparam = np.clip(
        np.sum(param_dirs * nlf_dirs, axis=-1), -1.0, 1.0
    )
    scores = np.sum(weights[None] * (1.0 - candidate_nlf), axis=1)
    chain_scale = float(np.sum(np.linalg.norm(
        candidates[0, list(ids)[1:]] - candidates[0, list(ids)[:-1]], axis=-1
    )))
    relative_depth = (
        candidates[:, list(ids)[1:], 2] - candidates[0, list(ids)[1:], 2]
    ) / max(chain_scale, 1e-12)
    ids_array = np.asarray(ids, dtype=np.int64)
    fit = np.linalg.norm(
        np.asarray(nlf_parametric_mm)[ids_array]
        - np.asarray(nlf_nonparametric_mm)[ids_array], axis=-1
    )
    uncertainty = np.asarray(nlf_uncertainty_mm)[ids_array]
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    common = np.asarray([
        float(side == "right"), float(uncertainty.mean()), float(fit.mean()),
        float(box[4]) if len(box) > 4 else 1.0,
        float(box[2] * box[3] / max(image_width * image_height, 1)),
    ])
    rows = []
    for index in range(len(candidates)):
        values = np.concatenate((
            np.asarray([scores[index], scores[0] - scores[index],
                        float(candidate_metrics[index, 4])]),
            common,
            candidate_nlf[index], h1_nlf, candidate_h1[index],
            relative_depth[index], candidate_dirs[index].reshape(-1),
            nlf_dirs.reshape(-1), weights, param_nonparam,
        ))
        rows.append(values)
    output = np.stack(rows).astype(np.float32)
    if output.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((output.shape, len(FEATURE_NAMES)))
    if not np.isfinite(output).all():
        raise ValueError("non-finite candidate feature")
    return output
