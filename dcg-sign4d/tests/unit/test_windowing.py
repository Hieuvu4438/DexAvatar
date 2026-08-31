from dataclasses import replace

import pytest
import torch

from dcg_sign4d.contact.ontology import VALID_FRAME_TRANSITIONS
from dcg_sign4d.contact.semi_markov import SemiMarkovDecoder
from dcg_sign4d.diffusion.state_codec import rotation_6d_to_matrix
from dcg_sign4d.inference.windowing import (
    slice_camera,
    slice_observations,
    slice_trajectory,
    stitch_contact_graphs,
    stitch_trajectories,
    window_starts,
)
from dcg_sign4d.initialization.camera import CameraTrajectory
from dcg_sign4d.synthetic import make_graph, make_observations, make_state


def test_window_plan_covers_tail_once():
    assert window_starts(10, 6, 2) == (0, 4)
    assert window_starts(11, 6, 2) == (0, 4, 5)
    assert window_starts(5, 6, 2) == (0,)


def test_camera_and_all_observation_cues_share_exact_window_boundaries():
    observations = make_observations(time=6)
    observations = replace(
        observations,
        part_masks=torch.ones(1, 6, 2, 3, 4),
        mask_reliability=torch.ones(1, 6, 2),
        tracks_2d=torch.ones(1, 6, 3, 2),
        track_reliability=torch.ones(1, 6, 3),
        depth_order=torch.ones(1, 6, 2),
        depth_reliability=torch.ones(1, 6, 2),
        metadata=(
            {
                "frame_ids": list(range(6)),
                "timestamps_sec": [index / 30 for index in range(6)],
            },
        ),
    ).validate()
    camera = CameraTrajectory(
        intrinsics=torch.eye(3)[None, None].expand(1, 6, 3, 3),
        world_to_camera=torch.eye(4)[None, None].expand(1, 6, 4, 4),
        image_size_wh=torch.ones(1, 6, 2) * 100,
        valid_mask=torch.ones(1, 6, dtype=torch.bool),
        coordinate_convention="test",
    ).validate()
    sliced_observations = slice_observations(observations, 2, 5)
    sliced_camera = slice_camera(camera, 2, 5)
    assert sliced_observations.keypoints_2d.shape[1] == 3
    assert sliced_observations.part_masks.shape[1] == 3
    assert sliced_observations.tracks_2d.shape[1] == 3
    assert sliced_observations.depth_order.shape[1] == 3
    assert sliced_observations.metadata[0]["frame_ids"] == [2, 3, 4]
    assert sliced_camera.intrinsics.shape[1] == 3


def test_rotation_aware_stitch_preserves_so3_and_clip_shape():
    state = make_state(time=8)
    starts = window_starts(8, 5, 2)
    windows = [slice_trajectory(state, start, start + 5) for start in starts]
    # Simulate a small independently sampled translation disagreement.
    windows[1] = replace(windows[1], root_translation=windows[1].root_translation + 0.01)
    stitched = stitch_trajectories(windows, starts, total_time=8, overlap=2)
    for field in ("root_rot6d", "body_rot6d", "left_hand_rot6d", "right_hand_rot6d"):
        matrix = rotation_6d_to_matrix(getattr(stitched, field))
        identity = matrix.transpose(-1, -2) @ matrix
        assert torch.allclose(identity, torch.eye(3), atol=1e-5)
        determinant = torch.linalg.det(matrix)
        assert torch.allclose(determinant, torch.ones_like(determinant), atol=1e-5)
    with pytest.raises(ValueError, match="clip-shared"):
        bad = [windows[0], replace(windows[1], beta=windows[1].beta + 1)]
        stitch_trajectories(bad, starts, total_time=8, overlap=2)


def test_contact_stitch_redecodes_valid_global_segments_in_seconds():
    starts = (0, 3)
    graphs = [make_graph(time=5), make_graph(time=5)]
    decoder = SemiMarkovDecoder(8, fps=20)
    graph = stitch_contact_graphs(
        graphs,
        starts,
        total_time=8,
        overlap=2,
        decoder=decoder,
        frame_valid=torch.ones(1, 8, dtype=torch.bool),
    )
    assert bool(VALID_FRAME_TRANSITIONS[graph.event_state[:, :-1], graph.event_state[:, 1:]].all())
    assert float(graph.segment_duration.max()) == pytest.approx(8 / 20)
