from pathlib import Path

import numpy as np

from signdart.geometry.arm_ik import (
    enumerate_body_pose_candidates,
    enumerate_three_link_body_pose_candidates,
)
from signdart.io.h1_state import H1State
from signdart.model import create_model, forward_state_batch


def test_h1_forward_and_candidate_wrist_compensation():
    state = H1State.load(Path(
        "/home/haipd/DexAvatar/SignEFT-X/runs/signeft_final_h1_full57/frames/Ablehnen/000149.npz"
    ))
    model = create_model(
        Path("/home/haipd/DexAvatar/SMPLer-X/common/utils/human_model_files"), "cpu"
    )
    vertices, joints = forward_state_batch(model, state, state.arrays["body_pose"], "cpu")
    max_error_mm = np.linalg.norm(vertices[0] - state.vertices_evaluator, axis=-1).max() * 1000.0
    assert max_error_mm <= 0.02
    parents = model.parents[:22].detach().cpu().numpy()
    candidates = enumerate_body_pose_candidates(
        state.arrays["global_orient"], state.arrays["body_pose"], parents,
        joints[0], state.K_evaluator, "left",
    )
    assert len(candidates) >= 2
    assert all(candidate.global_wrist_error_deg <= 0.01 for candidate in candidates)

    chain_candidates = enumerate_three_link_body_pose_candidates(
        state.arrays["global_orient"], state.arrays["body_pose"], parents,
        joints[0], state.K_evaluator, "left",
    )
    assert len(chain_candidates) >= len(candidates)
    assert all(candidate.global_wrist_error_deg <= 0.01 for candidate in chain_candidates)
