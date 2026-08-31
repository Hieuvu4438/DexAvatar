import pytest
import torch

from cusp_sl.geometry import axis_angle_to_matrix, geodesic_distance
from cusp_sl.retargeting import (
    compose_wrist_world,
    fuse_wilor_hand,
    mano_fingers_to_smplx,
    mirror_canonical_right_rotations,
    smplx_body_world_rotations,
    solve_elbow_for_wrist_alignment,
)


def test_left_chirality_conversion_is_an_involution_and_stays_in_so3():
    rotations = axis_angle_to_matrix(torch.randn(2, 15, 3) * 0.6)
    mirrored = mirror_canonical_right_rotations(rotations)
    round_trip = mirror_canonical_right_rotations(mirrored)
    identity = torch.eye(3).expand_as(mirrored)
    torch.testing.assert_close(round_trip, rotations)
    torch.testing.assert_close(mirrored.transpose(-1, -2) @ mirrored, identity)
    torch.testing.assert_close(torch.linalg.det(mirrored), torch.ones(2, 15))


def test_known_axis_chirality_conversion_matches_axis_angle_rule():
    axis_angle = torch.tensor([[0.20, -0.30, 0.40]])
    rotation = axis_angle_to_matrix(axis_angle)
    expected = axis_angle_to_matrix(axis_angle * torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(mirror_canonical_right_rotations(rotation), expected)


def test_right_hand_mapping_does_not_alias_or_change_input():
    rotations = axis_angle_to_matrix(torch.randn(15, 3) * 0.3)
    mapped = mano_fingers_to_smplx(rotations, is_right=True)
    torch.testing.assert_close(mapped, rotations)
    assert mapped.data_ptr() != rotations.data_ptr()


def test_elbow_solution_exactly_reaches_target_wrist_orientation():
    shoulder = axis_angle_to_matrix(torch.randn(8, 3) * 0.8)
    wrist_local = axis_angle_to_matrix(torch.randn(8, 3) * 0.5)
    target = axis_angle_to_matrix(torch.randn(8, 3) * 1.0)
    elbow = solve_elbow_for_wrist_alignment(shoulder, wrist_local, target)
    achieved = compose_wrist_world(shoulder, elbow, wrist_local)
    assert geodesic_distance(achieved, target).max() < 1e-5


def test_rest_pose_alignment_returns_identity_elbow():
    identity = torch.eye(3)
    elbow = solve_elbow_for_wrist_alignment(identity, identity, identity)
    torch.testing.assert_close(elbow, identity)


def test_adapter_rejects_incomplete_hand_pose_and_invalid_rotation_shape():
    with pytest.raises(ValueError, match="15 MANO"):
        mano_fingers_to_smplx(torch.eye(3).expand(14, 3, 3), is_right=True)
    with pytest.raises(ValueError, match="must end"):
        solve_elbow_for_wrist_alignment(torch.zeros(3), torch.eye(3), torch.eye(3))


def test_smplx_body_fk_uses_official_parent_chain():
    body = torch.eye(3).expand(2, 21, 3, 3).clone()
    root = axis_angle_to_matrix(torch.tensor([[0.2, -0.1, 0.3], [0.0, 0.4, 0.0]]))
    body[:, 11] = axis_angle_to_matrix(torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]))
    world = smplx_body_world_rotations(root, body)
    # Neck (body index 11/model joint 12) is downstream of spine joints, which
    # are identity in this fixture; its rotation composes with the pelvis.
    torch.testing.assert_close(world[:, 11], root @ body[:, 11])
    # Left shoulder branches from spine3 through left collar, not through neck.
    # Both rotations on that branch are identity in this fixture.
    torch.testing.assert_close(world[:, 15], root)


@pytest.mark.parametrize("is_right", [False, True])
def test_full_hand_fusion_reaches_wilor_wrist_and_preserves_unrelated_joints(is_right):
    base = axis_angle_to_matrix(torch.randn(51, 3) * 0.15)
    root = axis_angle_to_matrix(torch.randn(3) * 0.2)
    fingers = axis_angle_to_matrix(torch.randn(15, 3) * 0.4)
    target_canonical = axis_angle_to_matrix(torch.randn(3) * 0.7)
    original = base.clone()

    fused = fuse_wilor_hand(
        base, root, fingers, target_canonical, is_right=is_right
    )
    world = smplx_body_world_rotations(root, fused[:21])
    wrist_index = 20 if is_right else 19
    expected_target = (
        target_canonical
        if is_right
        else mirror_canonical_right_rotations(target_canonical)
    )
    assert geodesic_distance(world[wrist_index], expected_target) < 1e-5

    finger_start = 36 if is_right else 21
    expected_fingers = mano_fingers_to_smplx(fingers, is_right=is_right)
    torch.testing.assert_close(
        fused[finger_start : finger_start + 15], expected_fingers
    )
    changed_elbow = 18 if is_right else 17
    changed = {changed_elbow, *range(finger_start, finger_start + 15)}
    unchanged = [index for index in range(51) if index not in changed]
    torch.testing.assert_close(fused[unchanged], original[unchanged])
    torch.testing.assert_close(base, original)
