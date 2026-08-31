"""Small deterministic tensors used by smoke tests and CLI wiring checks."""

import torch

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


def make_state(batch: int = 1, time: int = 4) -> TrajectoryState:
    torch.manual_seed(10)
    return TrajectoryState(
        root_rot6d=torch.randn(batch, time, 6),
        root_translation=torch.randn(batch, time, 3),
        root_velocity=torch.randn(batch, time, 3),
        body_rot6d=torch.randn(batch, time, 2, 6),
        left_hand_rot6d=torch.randn(batch, time, 2, 6),
        right_hand_rot6d=torch.randn(batch, time, 2, 6),
        beta=torch.randn(batch, 3),
        valid_mask=torch.ones(batch, time, dtype=torch.bool),
    )


def make_observations(batch: int = 1, time: int = 4, joints: int = 3) -> ObservationBatch:
    return ObservationBatch(
        keypoints_2d=torch.randn(batch, time, joints, 2),
        keypoint_reliability=torch.full((batch, time, joints), 0.8),
        keypoint_valid=torch.ones(batch, time, joints, dtype=torch.bool),
        frame_valid=torch.ones(batch, time, dtype=torch.bool),
    )


def make_graph(batch: int = 1, time: int = 4, edges: int = 2) -> ContactGraphBatch:
    state = torch.zeros(batch, time, edges, dtype=torch.long)
    probability = torch.zeros(batch, time, edges, 4)
    probability[..., 0] = 1
    return ContactGraphBatch(
        event_state=state,
        event_probability=probability,
        edge_valid=torch.ones(batch, edges, dtype=torch.bool),
        uncertain_mask=torch.zeros(batch, time, edges, dtype=torch.bool),
        segment_id=torch.zeros(batch, time, edges, dtype=torch.long),
        segment_duration=torch.full((batch, time, edges), float(time)),
    )
