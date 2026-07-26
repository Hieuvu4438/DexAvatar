"""Realistic contiguous observation failures for Phase 2 training."""

from __future__ import annotations

import math

import torch

from phase2_refiner.data.dataset import (
    KEYPOINT_2D,
    KEYPOINT_2D_VALID,
    OBSERVATION_FEATURES,
    PALM_NORMAL,
    PALM_VALID,
    REPROJECTION_RESIDUAL_2D,
    ROTATION_6D,
    ROTATION_ACCELERATION,
    ROTATION_VELOCITY,
    TORSO_POSITION,
    TORSO_POSITION_VALID,
    U0_RELIABILITY,
    WRIST_POSITION,
    WRIST_POSITION_VALID,
)
from phase2_refiner.geometry.palm import FINGER_CHAINS
from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


REFINED_BODY = (2, 5, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20)
LEFT_HAND = tuple(range(21, 36))
RIGHT_HAND = tuple(range(36, 51))
DEFAULT_MODES = (
    "upper_body",
    "left_hand",
    "right_hand",
    "both_hands",
    "finger_chain",
    "wrist_attachment",
    "keypoint_dropout",
    "hand_swap",
    "crop_truncation",
)


def refresh_rotation_features(
    features: torch.Tensor, initial_matrix: torch.Tensor
) -> torch.Tensor:
    """Keep rotation, velocity, and acceleration tokens consistent with a pose."""
    features = features.clone()
    rot6d = matrix_to_rotation_6d(initial_matrix)
    features[..., ROTATION_6D] = rot6d
    features[..., ROTATION_VELOCITY] = 0.0
    features[..., ROTATION_ACCELERATION] = 0.0
    features[:, 1:, :, ROTATION_VELOCITY] = rot6d[:, 1:] - rot6d[:, :-1]
    features[:, 2:, :, ROTATION_ACCELERATION] = (
        features[:, 2:, :, ROTATION_VELOCITY]
        - features[:, 1:-1, :, ROTATION_VELOCITY]
    )
    return features


def _random_rotation_vectors(
    shape: tuple[int, ...], max_degrees: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    direction = torch.randn(shape, device=device, dtype=dtype)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    magnitude = torch.rand(shape[:-1] + (1,), device=device, dtype=dtype)
    return direction * magnitude * math.radians(max_degrees)


def _drop_observations(
    features: torch.Tensor, batch: int, frames: slice, joints: torch.Tensor
) -> None:
    selected = features[batch, frames]
    selected[:, joints, OBSERVATION_FEATURES.start] = 0.0
    selected[:, joints, OBSERVATION_FEATURES.start + 1] = 0.0
    selected[:, joints, OBSERVATION_FEATURES.start + 2] = 1.0
    selected[:, joints, KEYPOINT_2D] = 0.0
    selected[:, joints, KEYPOINT_2D_VALID] = 0.0
    selected[:, joints, U0_RELIABILITY] = 0.0
    selected[:, joints, TORSO_POSITION] = 0.0
    selected[:, joints, TORSO_POSITION_VALID] = 0.0
    selected[:, joints, WRIST_POSITION] = 0.0
    selected[:, joints, WRIST_POSITION_VALID] = 0.0
    selected[:, joints, PALM_NORMAL] = 0.0
    selected[:, joints, PALM_VALID] = 0.0


def _joint_indices(mode: str, device: torch.device) -> tuple[int, ...]:
    if mode == "upper_body":
        return REFINED_BODY
    if mode == "left_hand":
        return LEFT_HAND
    if mode == "right_hand":
        return RIGHT_HAND
    if mode == "both_hands":
        return LEFT_HAND + RIGHT_HAND
    if mode == "finger_chain":
        side_offset = 21 if int(torch.randint(0, 2, (), device=device)) == 0 else 36
        chains = tuple(FINGER_CHAINS.values())
        chain = chains[int(torch.randint(0, len(chains), (), device=device))]
        return tuple(side_offset + index for index in chain)
    if mode == "wrist_attachment":
        if int(torch.randint(0, 2, (), device=device)) == 0:
            return (19,) + LEFT_HAND
        return (20,) + RIGHT_HAND
    if mode in {"keypoint_dropout", "crop_truncation"}:
        return (
            LEFT_HAND
            if int(torch.randint(0, 2, (), device=device)) == 0
            else RIGHT_HAND
        )
    if mode == "hand_swap":
        return LEFT_HAND + RIGHT_HAND
    raise ValueError(f"Unknown corruption mode: {mode}")


def apply_burst_corruption(
    features: torch.Tensor,
    initial_matrix: torch.Tensor,
    frame_valid: torch.Tensor,
    target_rotation_valid: torch.Tensor | None = None,
    probability: float = 0.65,
    min_duration: int = 2,
    max_duration: int = 16,
    max_rotation_degrees: float = 35.0,
    modes: list[str] | tuple[str, ...] = DEFAULT_MODES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one realistic burst per selected clip while retaining clean batches."""
    if not modes:
        raise ValueError("At least one corruption mode is required")
    unknown = set(modes) - set(DEFAULT_MODES)
    if unknown:
        raise ValueError(f"Unknown corruption modes: {sorted(unknown)}")
    features = features.clone()
    initial_matrix = initial_matrix.clone()
    corruption_mask = torch.zeros(
        features.shape[:3], dtype=torch.bool, device=features.device
    )
    if target_rotation_valid is None:
        target_rotation_valid = frame_valid[:, :, None].expand(
            -1, -1, features.shape[2]
        )
    for batch_idx in range(features.shape[0]):
        if torch.rand((), device=features.device) > probability:
            continue
        valid_length = int(frame_valid[batch_idx].sum().item())
        if valid_length < min_duration:
            continue
        duration_limit = min(max_duration, valid_length)
        duration = int(
            torch.randint(min_duration, duration_limit + 1, (), device=features.device)
        )
        start = int(
            torch.randint(0, valid_length - duration + 1, (), device=features.device)
        )
        frames = slice(start, start + duration)
        candidates = []
        for candidate in modes:
            candidate_indices = _joint_indices(candidate, features.device)
            if target_rotation_valid[batch_idx, frames, list(candidate_indices)].any():
                candidates.append((candidate, candidate_indices))
        if not candidates:
            continue
        mode, indices = candidates[
            int(torch.randint(0, len(candidates), (), device=features.device))
        ]
        joints = torch.as_tensor(indices, device=features.device, dtype=torch.long)
        supervised = target_rotation_valid[batch_idx, frames][:, joints]
        corruption_mask[batch_idx, frames][:, joints] = supervised

        if mode == "hand_swap":
            # A swap is meaningful only when both hands are supervised throughout
            # the selected burst; otherwise fall back to a masked rotation burst.
            if not supervised.all():
                mode = "both_hands"
            else:
                left_features = features[batch_idx, frames, 21:36].clone()
                left_matrix = initial_matrix[batch_idx, frames, 21:36].clone()
                features[batch_idx, frames, 21:36] = features[batch_idx, frames, 36:51]
                features[batch_idx, frames, 36:51] = left_features
                initial_matrix[batch_idx, frames, 21:36] = initial_matrix[
                    batch_idx, frames, 36:51
                ]
                initial_matrix[batch_idx, frames, 36:51] = left_matrix
                features[batch_idx, frames, 21:51, OBSERVATION_FEATURES.start + 6] = 1.0
                features[batch_idx, frames, 21:51, U0_RELIABILITY] *= 0.25
                continue

        if mode == "crop_truncation":
            features[batch_idx, frames, joints, OBSERVATION_FEATURES.start + 4] = 1.0
            features[batch_idx, frames, joints, U0_RELIABILITY] *= 0.1
            features[batch_idx, frames, joints, KEYPOINT_2D_VALID] = 0.0
            features[batch_idx, frames, joints, KEYPOINT_2D] = 0.0
            continue

        _drop_observations(features, batch_idx, frames, joints)
        if mode != "keypoint_dropout":
            selected_matrix = initial_matrix[batch_idx, frames]
            current = selected_matrix[:, joints]
            residual = _random_rotation_vectors(
                current.shape[:-2] + (3,),
                max_rotation_degrees,
                current.device,
                current.dtype,
            )
            corrupted = axis_angle_to_matrix(residual) @ current
            selected_matrix[:, joints] = torch.where(
                supervised[..., None, None], corrupted, current
            )

    features = refresh_rotation_features(features, initial_matrix)
    return features, initial_matrix, corruption_mask


def apply_residual_mixture(
    features: torch.Tensor,
    initial_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    frame_valid: torch.Tensor,
    target_rotation_valid: torch.Tensor,
    *,
    real_fraction: float = 0.50,
    synthetic_fraction: float = 0.25,
    clean_fraction: float = 0.25,
    corruption: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the proposal's T2 real/synthetic/clean sample mixture.

    Mode IDs are 0=real expert residual, 1=synthetic burst from the clean
    target, and 2=clean identity. Invalid target joints retain the real
    initializer and cannot be synthetically corrupted.
    """
    fractions = torch.tensor(
        [real_fraction, synthetic_fraction, clean_fraction],
        dtype=torch.float64,
    )
    if (fractions < 0).any() or abs(float(fractions.sum()) - 1.0) > 1e-8:
        raise ValueError("T2 residual-mixture fractions must be non-negative and sum to 1")
    batch_size = features.shape[0]
    modes = torch.multinomial(
        fractions.to(device=features.device, dtype=torch.float32),
        batch_size,
        replacement=True,
    )
    output_features = features.clone()
    output_matrix = initial_matrix.clone()
    target_or_initial = torch.where(
        target_rotation_valid[..., None, None], target_matrix, initial_matrix
    )

    clean_or_synthetic = modes != 0
    if clean_or_synthetic.any():
        output_matrix[clean_or_synthetic] = target_or_initial[clean_or_synthetic]
        output_features[clean_or_synthetic] = refresh_rotation_features(
            output_features[clean_or_synthetic], output_matrix[clean_or_synthetic]
        )
        # Cached reprojection residuals are defined against the frozen real
        # initializer.  Once a mixture row is replaced by its target (clean or
        # synthetic-from-clean), those two channels no longer describe the
        # current pose.  Zero them rather than leak a contradictory H32 error.
        if output_features.shape[-1] >= REPROJECTION_RESIDUAL_2D.stop:
            output_features[
                clean_or_synthetic, :, :, REPROJECTION_RESIDUAL_2D
            ] = 0.0

    corruption_mask = torch.zeros_like(target_rotation_valid)
    synthetic = modes == 1
    if synthetic.any():
        settings = dict(corruption or {})
        settings["probability"] = 1.0
        synthetic_features, synthetic_matrix, synthetic_mask = apply_burst_corruption(
            output_features[synthetic],
            output_matrix[synthetic],
            frame_valid[synthetic],
            target_rotation_valid=target_rotation_valid[synthetic],
            **settings,
        )
        output_features[synthetic] = synthetic_features
        output_matrix[synthetic] = synthetic_matrix
        corruption_mask[synthetic] = synthetic_mask
    return output_features, output_matrix, corruption_mask, modes
