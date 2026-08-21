from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..data.cache import ObservationBatch
from ..io.predictions import PredictionArtifact
from ..utils.hashing import sha256_file


def _summary(values: np.ndarray) -> tuple[list[float], list[str]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0] * 6, ["mean", "std", "max", "q25", "q50", "q75"]
    return [
        float(values.mean()),
        float(values.std()),
        float(values.max()),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.75)),
    ], ["mean", "std", "max", "q25", "q50", "q75"]


def _motion(joints: np.ndarray) -> np.ndarray:
    result = np.zeros(len(joints), dtype=np.float64)
    if len(joints) > 1:
        result[1:] = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(-1)
        result[0] = result[1]
    return result


def _append_stats(
    values: list[float], names: list[str], prefix: str, array: np.ndarray
) -> None:
    stats, suffixes = _summary(array)
    values.extend(stats)
    names.extend(f"{prefix}_{suffix}" for suffix in suffixes)


def extract_gate_features(
    candidate: PredictionArtifact,
    baseline: PredictionArtifact,
    observations: ObservationBatch,
    diagnostics: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Build GT-free, clip-local features for selecting a coherent frame hypothesis."""
    frame_ids = candidate.frame_ids.tolist()
    if frame_ids != baseline.frame_ids.tolist() or frame_ids != observations.frame_ids.tolist():
        raise ValueError("gate inputs must have identical frame IDs")
    if candidate.rotations is None or baseline.rotations is None:
        raise ValueError("gate requires rotation-complete candidate and baseline artifacts")
    if observations.joints_3d.shape[1] < 3:
        raise ValueError("gate requires body, hand, and legacy observation hypotheses")
    change = np.asarray(diagnostics.get("change_probability"), dtype=np.float64)
    if change.shape != (len(frame_ids),):
        raise ValueError("candidate diagnostics have incompatible change_probability")

    candidate_joints = candidate.joints_3d.detach().cpu().numpy()
    baseline_joints = baseline.joints_3d.detach().cpu().numpy()
    observed_joints = observations.joints_3d.detach().cpu().numpy()
    valid = observations.valid_3d.detach().cpu().numpy()
    observation_features = observations.features.detach().cpu().numpy()
    candidate_risk = candidate.risk_score.detach().cpu().numpy()
    baseline_risk = baseline.risk_score.detach().cpu().numpy()
    uncertainty = candidate.uncertainty.detach().cpu().numpy()
    abstain = candidate.abstain.detach().cpu().numpy()
    candidate_motion = _motion(candidate_joints[:, 25:40])
    baseline_motion = _motion(baseline_joints[:, 25:40])
    delta_motion = _motion(candidate_joints[:, 25:40] - baseline_joints[:, 25:40])
    joint_delta = candidate_joints[:, 25:40] - baseline_joints[:, 25:40]
    joint_delta -= joint_delta.mean(1, keepdims=True)
    joint_delta = np.linalg.norm(joint_delta, axis=-1)
    rotation_delta = np.linalg.norm(
        candidate.rotations[:, 25:40].detach().cpu().numpy()
        - baseline.rotations[:, 25:40].detach().cpu().numpy(),
        axis=(2, 3),
    )

    rows: list[list[float]] = []
    feature_names: list[str] | None = None
    for frame in range(len(frame_ids)):
        values: list[float] = []
        names: list[str] = []

        _append_stats(values, names, "candidate_baseline_joint", joint_delta[frame])
        for source_a, source_b, prefix in (
            (1, 2, "wilor_legacy"),
            (0, 2, "smplerx_legacy"),
            (0, 1, "smplerx_wilor"),
        ):
            mask = valid[frame, source_a, 25:40] & valid[frame, source_b, 25:40]
            discrepancy = np.linalg.norm(
                observed_joints[frame, source_a, 25:40]
                - observed_joints[frame, source_b, 25:40],
                axis=-1,
            )
            _append_stats(values, names, prefix, discrepancy[mask])
            values.append(float(mask.mean()))
            names.append(f"{prefix}_valid_fraction")
        _append_stats(values, names, "candidate_uncertainty_left", uncertainty[frame, 25:40])
        _append_stats(values, names, "candidate_uncertainty_body", uncertainty[frame, :25])
        _append_stats(values, names, "candidate_uncertainty_right", uncertainty[frame, 40:55])
        _append_stats(values, names, "candidate_baseline_rotation", rotation_delta[frame])
        scalar_features = (
            ("candidate_risk_left", candidate_risk[frame, 1]),
            ("candidate_risk_body", candidate_risk[frame, 0]),
            ("candidate_risk_right", candidate_risk[frame, 2]),
            ("baseline_risk_left", baseline_risk[frame, 1]),
            ("change_probability", change[frame]),
            ("candidate_motion", candidate_motion[frame]),
            ("baseline_motion", baseline_motion[frame]),
            ("delta_motion", delta_motion[frame]),
            ("time_normalized", frame / max(1, len(frame_ids) - 1)),
            ("clip_length_normalized", len(frame_ids) / 64.0),
            ("candidate_abstain_left", abstain[frame, 1]),
        )
        for name, value in scalar_features:
            names.append(name)
            values.append(float(value))
        for source in range(3):
            for feature in range(min(6, observation_features.shape[-1])):
                names.append(f"observation_source_{source}_feature_{feature}_left_mean")
                values.append(float(observation_features[frame, source, 25:40, feature].mean()))
        for offset in (-2, -1, 1, 2):
            neighbor = min(len(frame_ids) - 1, max(0, frame + offset))
            for name, value in (
                ("risk_left", candidate_risk[neighbor, 1]),
                ("change", change[neighbor]),
                ("joint_delta", joint_delta[neighbor].mean()),
                ("candidate_motion", candidate_motion[neighbor]),
                ("baseline_motion", baseline_motion[neighbor]),
            ):
                names.append(f"neighbor_{offset:+d}_{name}")
                values.append(float(value))
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise AssertionError("gate feature schema changed between frames")
        rows.append(values)

    local = np.asarray(rows, dtype=np.float32)
    clip_summary = np.concatenate(
        (
            local.mean(0),
            local.std(0),
            np.quantile(local, 0.25, axis=0),
            np.quantile(local, 0.75, axis=0),
        )
    ).astype(np.float32)
    summary_names = [
        f"clip_{stat}_{name}"
        for stat in ("mean", "std", "q25", "q75")
        for name in feature_names or []
    ]
    matrix = np.concatenate((local, np.repeat(clip_summary[None], len(local), axis=0)), axis=1)
    names = (feature_names or []) + summary_names
    if not np.isfinite(matrix).all():
        raise ValueError("gate features contain non-finite values")
    return matrix, names


@dataclass(frozen=True)
class ExtraTreesArtifact:
    children_left: torch.Tensor
    children_right: torch.Tensor
    feature: torch.Tensor
    threshold: torch.Tensor
    value: torch.Tensor
    node_count: torch.Tensor
    feature_names: tuple[str, ...]
    decision_threshold_mm: float
    switch_penalty_mm: float
    metadata: dict[str, Any]

    @classmethod
    def from_sklearn(
        cls,
        estimator: Any,
        feature_names: list[str],
        decision_threshold_mm: float,
        switch_penalty_mm: float,
        metadata: dict[str, Any],
    ) -> ExtraTreesArtifact:
        trees = [item.tree_ for item in estimator.estimators_]
        maximum = max(tree.node_count for tree in trees)
        count = len(trees)
        children_left = torch.full((count, maximum), -1, dtype=torch.int64)
        children_right = torch.full_like(children_left, -1)
        feature = torch.full_like(children_left, -2)
        threshold = torch.zeros((count, maximum), dtype=torch.float64)
        value = torch.zeros((count, maximum), dtype=torch.float64)
        node_count = torch.zeros(count, dtype=torch.int64)
        for index, tree in enumerate(trees):
            nodes = tree.node_count
            node_count[index] = nodes
            children_left[index, :nodes] = torch.from_numpy(tree.children_left.astype(np.int64))
            children_right[index, :nodes] = torch.from_numpy(tree.children_right.astype(np.int64))
            feature[index, :nodes] = torch.from_numpy(tree.feature.astype(np.int64))
            threshold[index, :nodes] = torch.from_numpy(tree.threshold.astype(np.float64))
            value[index, :nodes] = torch.from_numpy(tree.value[:, 0, 0].astype(np.float64))
        return cls(
            children_left,
            children_right,
            feature,
            threshold,
            value,
            node_count,
            tuple(feature_names),
            float(decision_threshold_mm),
            float(switch_penalty_mm),
            metadata,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("gate feature matrix does not match the frozen schema")
        prediction = np.zeros(features.shape[0], dtype=np.float64)
        left = self.children_left.numpy()
        right = self.children_right.numpy()
        split_feature = self.feature.numpy()
        threshold = self.threshold.numpy()
        leaf_value = self.value.numpy()
        for tree in range(len(self.node_count)):
            nodes = np.zeros(features.shape[0], dtype=np.int64)
            active = left[tree, nodes] != -1
            while active.any():
                rows = np.flatnonzero(active)
                current = nodes[rows]
                columns = split_feature[tree, current]
                go_left = features[rows, columns] <= threshold[tree, current]
                nodes[rows] = np.where(
                    go_left, left[tree, current], right[tree, current]
                )
                active = left[tree, nodes] != -1
            prediction += leaf_value[tree, nodes]
        return prediction / len(self.node_count)

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        tensor_path = root / "forest.safetensors"
        save_file(
            {
                "children_left": self.children_left,
                "children_right": self.children_right,
                "feature": self.feature,
                "threshold": self.threshold,
                "value": self.value,
                "node_count": self.node_count,
            },
            tensor_path,
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "schema_version": "1.0",
                "model_type": "extra_trees_regressor_safe",
                "feature_names": list(self.feature_names),
                "decision_threshold_mm": self.decision_threshold_mm,
                "switch_penalty_mm": self.switch_penalty_mm,
                "forest_sha256": sha256_file(tensor_path),
            }
        )
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, root: str | Path) -> ExtraTreesArtifact:
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        tensor_path = root / "forest.safetensors"
        if sha256_file(tensor_path) != metadata["forest_sha256"]:
            raise ValueError("gate forest hash mismatch")
        values = load_file(tensor_path)
        return cls(
            children_left=values["children_left"],
            children_right=values["children_right"],
            feature=values["feature"],
            threshold=values["threshold"],
            value=values["value"],
            node_count=values["node_count"],
            feature_names=tuple(metadata["feature_names"]),
            decision_threshold_mm=float(metadata["decision_threshold_mm"]),
            switch_penalty_mm=float(metadata["switch_penalty_mm"]),
            metadata=metadata,
        )


def decode_gate_sequence(
    predicted_delta_mm: np.ndarray,
    decision_threshold_mm: float,
    switch_penalty_mm: float,
) -> np.ndarray:
    """Viterbi-decode baseline/candidate states with a fixed switching cost."""
    predicted = np.asarray(predicted_delta_mm, dtype=np.float64)
    if predicted.ndim != 1 or not np.isfinite(predicted).all():
        raise ValueError("predicted gate deltas must be finite [T]")
    if switch_penalty_mm < 0:
        raise ValueError("switch penalty must be non-negative")
    if not len(predicted):
        return np.zeros(0, dtype=bool)
    costs = np.zeros((len(predicted), 2), dtype=np.float64)
    back = np.zeros((len(predicted), 2), dtype=np.int64)
    costs[0, 1] = predicted[0] - decision_threshold_mm
    for frame in range(1, len(predicted)):
        emission = (0.0, predicted[frame] - decision_threshold_mm)
        for state in (0, 1):
            options = costs[frame - 1] + switch_penalty_mm * (
                np.arange(2) != state
            )
            previous = int(np.argmin(options))
            back[frame, state] = previous
            costs[frame, state] = options[previous] + emission[state]
    states = np.zeros(len(predicted), dtype=np.int64)
    states[-1] = int(np.argmin(costs[-1]))
    for frame in range(len(predicted) - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return states.astype(bool)


def decode_multigate_sequence(emission_cost_mm: np.ndarray, switch_penalty_mm: float) -> np.ndarray:
    """Viterbi-decode one of K coherent hypotheses from a [T,K] cost matrix."""
    emission = np.asarray(emission_cost_mm, dtype=np.float64)
    if emission.ndim != 2 or not len(emission) or not np.isfinite(emission).all():
        raise ValueError("multi-gate emission costs must be finite [T,K]")
    if switch_penalty_mm < 0:
        raise ValueError("switch penalty must be non-negative")
    frames, states_count = emission.shape
    costs = np.empty((frames, states_count), dtype=np.float64)
    back = np.zeros((frames, states_count), dtype=np.int64)
    costs[0] = emission[0]
    state_ids = np.arange(states_count)
    for frame in range(1, frames):
        for state in range(states_count):
            options = costs[frame - 1] + switch_penalty_mm * (state_ids != state)
            previous = int(np.argmin(options))
            back[frame, state] = previous
            costs[frame, state] = options[previous] + emission[frame, state]
    states = np.zeros(frames, dtype=np.int64)
    states[-1] = int(np.argmin(costs[-1]))
    for frame in range(frames - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return states


def merge_predictions(
    candidate: PredictionArtifact, baseline: PredictionArtifact, use_candidate: np.ndarray
) -> PredictionArtifact:
    if candidate.frame_ids.tolist() != baseline.frame_ids.tolist():
        raise ValueError("candidate and baseline frame IDs differ")
    selection = torch.as_tensor(use_candidate, dtype=torch.bool)
    if selection.shape != candidate.frame_ids.shape:
        raise ValueError("gate selection must have shape [T]")

    def choose(candidate_value: torch.Tensor | None, baseline_value: torch.Tensor | None):
        if candidate_value is None and baseline_value is None:
            return None
        if candidate_value is None or baseline_value is None:
            raise ValueError("candidate and baseline optional tensor contracts differ")
        mask = selection.reshape((-1,) + (1,) * (candidate_value.ndim - 1))
        return torch.where(mask, candidate_value.cpu(), baseline_value.cpu())

    merged = PredictionArtifact(
        frame_ids=candidate.frame_ids.cpu(),
        joints_3d=choose(candidate.joints_3d, baseline.joints_3d),
        rotations=choose(candidate.rotations, baseline.rotations),
        translation=choose(candidate.translation, baseline.translation),
        vertices=choose(candidate.vertices, baseline.vertices),
        risk_score=choose(candidate.risk_score, baseline.risk_score),
        abstain=choose(candidate.abstain, baseline.abstain),
        uncertainty=choose(candidate.uncertainty, baseline.uncertainty),
        contact_probability=choose(candidate.contact_probability, baseline.contact_probability),
        contacts=choose(candidate.contacts, baseline.contacts),
    )
    merged.validate()
    return merged


def merge_multiple_predictions(
    hypotheses: list[PredictionArtifact], states: np.ndarray
) -> PredictionArtifact:
    if not hypotheses:
        raise ValueError("at least one prediction hypothesis is required")
    frame_ids = hypotheses[0].frame_ids.tolist()
    if any(item.frame_ids.tolist() != frame_ids for item in hypotheses[1:]):
        raise ValueError("multi-gate hypotheses have different frame IDs")
    selection = torch.as_tensor(states, dtype=torch.int64)
    if selection.shape != hypotheses[0].frame_ids.shape:
        raise ValueError("multi-gate states must have shape [T]")
    if int(selection.min()) < 0 or int(selection.max()) >= len(hypotheses):
        raise ValueError("multi-gate state index is outside the hypothesis set")

    def choose(name: str) -> torch.Tensor | None:
        tensors = [getattr(item, name) for item in hypotheses]
        if all(value is None for value in tensors):
            return None
        if any(value is None for value in tensors):
            raise ValueError(f"multi-gate optional tensor contract differs for {name}")
        result = tensors[0].detach().cpu().clone()  # type: ignore[union-attr]
        for state, value in enumerate(tensors[1:], start=1):
            mask = selection == state
            result[mask] = value.detach().cpu()[mask]  # type: ignore[union-attr]
        return result

    merged = PredictionArtifact(
        frame_ids=hypotheses[0].frame_ids.cpu(),
        joints_3d=choose("joints_3d"),
        rotations=choose("rotations"),
        translation=choose("translation"),
        vertices=choose("vertices"),
        risk_score=choose("risk_score"),
        abstain=choose("abstain"),
        uncertainty=choose("uncertainty"),
        contact_probability=choose("contact_probability"),
        contacts=choose("contacts"),
    )
    merged.validate()
    return merged
