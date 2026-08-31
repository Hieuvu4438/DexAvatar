"""Alignment-explicit hand placement and articulation endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IndexArray = NDArray[np.integer]


def point_error(source: FloatArray, target: FloatArray) -> FloatArray:
    if source.shape != target.shape or source.shape[-1] != 3:
        raise ValueError("source/target points must have equal [...,3] shape")
    return np.linalg.norm(source - target, axis=-1)


def aligned_vertex_error(
    source: FloatArray,
    target: FloatArray,
    source_center: FloatArray,
    target_center: FloatArray,
) -> FloatArray:
    return point_error(source - source_center, target - target_center)


@dataclass(frozen=True)
class HandPlacementMetrics:
    joint_regressor: FloatArray
    left_hand_ids: IndexArray
    right_hand_ids: IndexArray
    body_ids: IndexArray
    left_wrist_index: int = 20
    right_wrist_index: int = 21
    pelvis_indices: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        if self.joint_regressor.ndim != 2:
            raise ValueError("joint regressor must be [J,V]")
        vertex_count = self.joint_regressor.shape[1]
        for indices in (self.left_hand_ids, self.right_hand_ids, self.body_ids):
            if indices.ndim != 1 or len(indices) == 0:
                raise ValueError("region indices must be non-empty vectors")
            if indices.min() < 0 or indices.max() >= vertex_count:
                raise ValueError("region index outside model topology")

    def evaluate_frame(self, source: FloatArray, target: FloatArray) -> dict[str, float]:
        vertex_count = self.joint_regressor.shape[1]
        if source.shape != (vertex_count, 3) or target.shape != source.shape:
            raise ValueError(f"vertices must be [{vertex_count},3]")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("vertices contain NaN/Inf")
        source_joints = self.joint_regressor @ source
        target_joints = self.joint_regressor @ target
        source_root = source_joints[list(self.pelvis_indices)].mean(0, keepdims=True)
        target_root = target_joints[list(self.pelvis_indices)].mean(0, keepdims=True)
        output: dict[str, float] = {}
        for side, indices, wrist_index in (
            ("left", self.left_hand_ids, self.left_wrist_index),
            ("right", self.right_hand_ids, self.right_wrist_index),
        ):
            source_hand = source[indices]
            target_hand = target[indices]
            output[f"root_aligned_{side}_hand_pve_mm"] = float(
                aligned_vertex_error(source_hand, target_hand, source_root, target_root).mean()
                * 1000
            )
            output[f"wrist_aligned_{side}_hand_pve_mm"] = float(
                aligned_vertex_error(
                    source_hand,
                    target_hand,
                    source_joints[wrist_index][None],
                    target_joints[wrist_index][None],
                ).mean()
                * 1000
            )
            # Attached-author TR metric: each hand region removes its own centroid.
            output[f"legacy_region_tr_{side}_hand_pve_mm"] = float(
                aligned_vertex_error(
                    source_hand,
                    target_hand,
                    source_hand.mean(0, keepdims=True),
                    target_hand.mean(0, keepdims=True),
                ).mean()
                * 1000
            )
        output["root_aligned_body_pve_mm"] = float(
            aligned_vertex_error(
                source[self.body_ids], target[self.body_ids], source_root, target_root
            ).mean()
            * 1000
        )
        body_joint_ids = np.flatnonzero(
            np.abs(self.joint_regressor[: min(22, len(source_joints))]).sum(axis=1) > 0
        )
        output["root_aligned_body_mpjpe_mm"] = float(
            aligned_vertex_error(
                source_joints[body_joint_ids],
                target_joints[body_joint_ids],
                source_root,
                target_root,
            ).mean()
            * 1000
        )
        return output
