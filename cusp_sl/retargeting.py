"""Testable MANO-to-SMPL-X rotation and wrist-alignment primitives.

These functions implement the coordinate contract required by CUSP-SL's A1
frontend. They do not claim to reproduce Tamaththul3D's unreleased code. The
caller must first express the WiLoR wrist target and SMPL-X kinematic chain in
the same camera/world basis. SMPL-X must be decoded with ``flat_hand_mean=True``
when the returned finger rotations are used directly.
"""

from __future__ import annotations

import torch


# SMPL-X model-joint parents for pelvis (0) and the 21 body-pose joints.  Cache
# body index ``i`` corresponds to model joint ``i + 1``.  Keeping this table
# explicit makes the A1 adapter independent of a particular renderer package.
SMPLX_BODY_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
)


def _require_rotations(value: torch.Tensor, name: str) -> None:
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must end in (3, 3), got {tuple(value.shape)}")


def mirror_canonical_right_rotations(rotations: torch.Tensor) -> torch.Tensor:
    """Undo WiLoR's canonical-right convention for a detected left hand.

    Reflection across the YZ plane is applied as ``M R M^T`` with
    ``M = diag(-1, 1, 1)``. Although ``M`` is an improper rotation, the
    conjugated result remains in SO(3). Applying this function twice is exact
    up to floating-point roundoff.
    """

    _require_rotations(rotations, "rotations")
    signs = rotations.new_tensor([-1.0, 1.0, 1.0])
    return rotations * signs[..., :, None] * signs[..., None, :]


def mano_fingers_to_smplx(
    rotations: torch.Tensor, *, is_right: bool
) -> torch.Tensor:
    """Map WiLoR's 15 MANO local rotations to flat-mean SMPL-X parameters."""

    _require_rotations(rotations, "rotations")
    if rotations.shape[-3] != 15:
        raise ValueError(f"Expected 15 MANO finger joints, got {rotations.shape[-3]}")
    return rotations.clone() if is_right else mirror_canonical_right_rotations(rotations)


def solve_elbow_for_wrist_alignment(
    shoulder_world: torch.Tensor,
    wrist_local: torch.Tensor,
    target_wrist_world: torch.Tensor,
) -> torch.Tensor:
    """Solve the local elbow rotation for an exact target wrist orientation.

    With the chain ``R_wrist_world = R_shoulder_world R_elbow_local
    R_wrist_local``, the returned rotation is the unique SO(3) solution. This
    is the closed-form geometric core described by Tamaththul3D Eq. (4), kept
    separate from any optional shoulder optimization or swing-twist policy.
    """

    _require_rotations(shoulder_world, "shoulder_world")
    _require_rotations(wrist_local, "wrist_local")
    _require_rotations(target_wrist_world, "target_wrist_world")
    try:
        torch.broadcast_shapes(
            shoulder_world.shape[:-2],
            wrist_local.shape[:-2],
            target_wrist_world.shape[:-2],
        )
    except RuntimeError as error:
        raise ValueError("rotation batch dimensions are not broadcast-compatible") from error
    return (
        shoulder_world.transpose(-1, -2)
        @ target_wrist_world
        @ wrist_local.transpose(-1, -2)
    )


def compose_wrist_world(
    shoulder_world: torch.Tensor,
    elbow_local: torch.Tensor,
    wrist_local: torch.Tensor,
) -> torch.Tensor:
    """Forward-kinematic wrist orientation for adapter validation."""

    _require_rotations(shoulder_world, "shoulder_world")
    _require_rotations(elbow_local, "elbow_local")
    _require_rotations(wrist_local, "wrist_local")
    return shoulder_world @ elbow_local @ wrist_local


def smplx_body_world_rotations(
    global_orient: torch.Tensor, body_rotations: torch.Tensor
) -> torch.Tensor:
    """Compose the 21 SMPL-X body rotations in the model/world basis.

    ``global_orient`` is the pelvis rotation and ``body_rotations`` follows the
    official SMPL-X body-pose order.  The returned tensor has the same leading
    batch dimensions and 21 rotations; it does not include the pelvis.
    """

    _require_rotations(global_orient, "global_orient")
    _require_rotations(body_rotations, "body_rotations")
    if body_rotations.shape[-3] != 21:
        raise ValueError(
            f"Expected 21 SMPL-X body joints, got {body_rotations.shape[-3]}"
        )
    try:
        torch.broadcast_shapes(global_orient.shape[:-2], body_rotations.shape[:-3])
    except RuntimeError as error:
        raise ValueError("body/global batch dimensions are not broadcast-compatible") from error

    model_world = [global_orient]
    for model_joint in range(1, 22):
        parent = SMPLX_BODY_PARENTS[model_joint]
        model_world.append(
            model_world[parent] @ body_rotations[..., model_joint - 1, :, :]
        )
    return torch.stack(model_world[1:], dim=-3)


def fuse_wilor_hand(
    base_rotations: torch.Tensor,
    global_orient: torch.Tensor,
    mano_finger_rotations: torch.Tensor,
    mano_global_orient: torch.Tensor,
    *,
    is_right: bool,
) -> torch.Tensor:
    """Fuse one WiLoR hand into a 51-joint SMPL-X cache pose.

    The WiLoR global wrist orientation and the SMPL-X chain must already use
    the same camera/world basis.  WiLoR inference crops do not rotate the image;
    their only orientation-changing operation is the documented horizontal
    flip for left hands, which is undone here.  The function replaces the 15
    finger rotations and solves the elbow rotation from Tamaththul3D Eq. (4),
    while preserving the shoulder, wrist-local rotation, root, shape, camera,
    and every unrelated joint.
    """

    _require_rotations(base_rotations, "base_rotations")
    if base_rotations.shape[-3] != 51:
        raise ValueError(f"Expected 51 cache joints, got {base_rotations.shape[-3]}")
    _require_rotations(mano_global_orient, "mano_global_orient")
    fingers = mano_fingers_to_smplx(mano_finger_rotations, is_right=is_right)
    target_wrist = (
        mano_global_orient
        if is_right
        else mirror_canonical_right_rotations(mano_global_orient)
    )

    output = base_rotations.clone()
    body_world = smplx_body_world_rotations(global_orient, output[..., :21, :, :])
    shoulder_index = 16 if is_right else 15
    elbow_index = 18 if is_right else 17
    wrist_index = 20 if is_right else 19
    finger_start = 36 if is_right else 21
    output[..., elbow_index, :, :] = solve_elbow_for_wrist_alignment(
        body_world[..., shoulder_index, :, :],
        output[..., wrist_index, :, :],
        target_wrist,
    )
    output[..., finger_start : finger_start + 15, :, :] = fingers
    return output


__all__ = [
    "SMPLX_BODY_PARENTS",
    "compose_wrist_world",
    "fuse_wilor_hand",
    "mano_fingers_to_smplx",
    "mirror_canonical_right_rotations",
    "smplx_body_world_rotations",
    "solve_elbow_for_wrist_alignment",
]
