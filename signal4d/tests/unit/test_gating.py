from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from signal4d.models.gating import (
    ExtraTreesArtifact,
    decode_gate_sequence,
    decode_multigate_sequence,
)
from signal4d.optimization.state import SequenceState


def test_gate_sequence_switch_penalty_removes_single_frame_island() -> None:
    predicted = np.asarray([1.0, 1.0, -0.1, 1.0, 1.0])
    without_penalty = decode_gate_sequence(predicted, 0.0, 0.0)
    with_penalty = decode_gate_sequence(predicted, 0.0, 0.2)
    assert without_penalty.tolist() == [False, False, True, False, False]
    assert with_penalty.tolist() == [False] * 5


def test_safe_extra_trees_roundtrip_matches_sklearn(tmp_path) -> None:
    generator = np.random.default_rng(12345)
    features = generator.normal(size=(80, 4))
    target = features[:, 0] ** 2 - 0.5 * features[:, 1]
    model = ExtraTreesRegressor(
        n_estimators=11,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=12345,
    ).fit(features, target)
    artifact = ExtraTreesArtifact.from_sklearn(
        model,
        ["a", "b", "c", "d"],
        decision_threshold_mm=0.0,
        switch_penalty_mm=0.5,
        metadata={"purpose": "test"},
    )
    artifact.save(tmp_path)
    restored = ExtraTreesArtifact.load(tmp_path)
    np.testing.assert_allclose(restored.predict(features), model.predict(features), atol=1e-12)


def test_multigate_decodes_lowest_coherent_hypothesis() -> None:
    emission = np.asarray(
        [
            [0.0, -1.0, 0.5],
            [0.0, -1.0, 0.5],
            [0.0, 2.0, -0.1],
            [0.0, -1.0, 0.5],
        ]
    )
    assert decode_multigate_sequence(emission, 0.0).tolist() == [1, 1, 2, 1]
    assert decode_multigate_sequence(emission, 2.0).tolist() == [1, 1, 1, 1]


def test_sequence_state_accepts_per_frame_shape() -> None:
    state = SequenceState(
        global_rot6d=torch.zeros((2, 6)),
        body_rot6d=torch.zeros((2, 21, 6)),
        left_hand_rot6d=torch.zeros((2, 15, 6)),
        right_hand_rot6d=torch.zeros((2, 15, 6)),
        translation=torch.zeros((2, 3)),
        betas=torch.zeros((2, 10)),
    )
    state.validate()
