from __future__ import annotations

import numpy as np
import torch

from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from signal4d_external.hand_v2_core import (
    exact_rank_selection,
    geodesic_blend,
    hand_eligibility,
    selected_count,
    smooth_scores,
)


def test_geodesic_blend_endpoints_and_halfway() -> None:
    initial = torch.eye(3)[None]
    candidate = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, 0.8]]))
    torch.testing.assert_close(geodesic_blend(initial, candidate, 0.0), initial)
    torch.testing.assert_close(geodesic_blend(initial, candidate, 1.0), candidate)
    distance = geodesic_distance(geodesic_blend(initial, candidate, 0.5), initial)
    torch.testing.assert_close(distance, torch.tensor([0.4]))


def test_exact_rank_selection_is_global_and_stable_on_ties() -> None:
    selected = exact_rank_selection(
        [np.asarray([0.5, 0.9]), np.asarray([0.9, 0.1])], 0.5
    )
    np.testing.assert_array_equal(selected[0], [False, True])
    np.testing.assert_array_equal(selected[1], [True, False])
    assert sum(value.sum() for value in selected) == selected_count(0.5, 4)


def test_exact_rank_selection_never_selects_ineligible_frames() -> None:
    selected = exact_rank_selection(
        [np.asarray([100.0, 0.9]), np.asarray([0.8, 0.7])],
        0.5,
        [np.asarray([False, True]), np.asarray([True, True])],
    )
    np.testing.assert_array_equal(selected[0], [False, True])
    np.testing.assert_array_equal(selected[1], [True, False])


def test_smooth_scores_does_not_cross_clip_boundaries() -> None:
    scores = np.asarray([0.0, 1.0, 0.0])
    timestamps = np.asarray([0.0, 0.1, 0.2])
    np.testing.assert_allclose(
        smooth_scores(scores, timestamps, 0.11), [0.5, 1 / 3, 0.5]
    )
    np.testing.assert_allclose(
        smooth_scores(np.asarray([1.0]), np.asarray([5.0]), 2.0), [1.0]
    )


def test_smooth_scores_uses_time_not_index_distance() -> None:
    scores = np.asarray([0.0, 1.0, 0.0])
    sparse = np.asarray([0.0, 0.1, 5.0])
    np.testing.assert_allclose(smooth_scores(scores, sparse, 0.11), [0.5, 0.5, 0.0])


def test_hand_eligibility_requires_visibility_and_reliability() -> None:
    class Clip:
        keypoint_valid = np.ones((3, 51), dtype=bool)
        u0_reliability = np.ones((3, 51), dtype=np.float32)

    clip = Clip()
    clip.keypoint_valid[0, 21:36] = False
    clip.u0_reliability[1, 21:36] = 0.1
    np.testing.assert_array_equal(hand_eligibility(clip, "lhand"), [False, False, True])
