from __future__ import annotations

# SMPL-X body_pose order. Global orientation is deliberately not part of this map.
BODY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

_BODY_JOINT_INDEX = {name: index for index, name in enumerate(BODY_JOINT_NAMES)}


def body_joint_indices(names: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    """Resolve joint names once and reject duplicates/implicit raw-index contracts."""
    unknown = sorted(set(names) - _BODY_JOINT_INDEX.keys())
    if unknown:
        raise ValueError(f"unknown SMPL-X body joints: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("open_body_joints must not contain duplicates")
    return tuple(_BODY_JOINT_INDEX[name] for name in names)

