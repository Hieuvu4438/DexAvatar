"""Padded whole-sequence dataset backed by versioned Phase 2 caches."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from phase2_refiner.data.cache_schema import NUM_JOINTS, load_cache_clip
from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from phase2_refiner.geometry.palm import MCP_INDICES, palm_normal


ROTATION_6D = slice(0, 6)
ROTATION_VELOCITY = slice(6, 12)
ROTATION_ACCELERATION = slice(12, 18)
OBSERVATION_FEATURES = slice(18, 26)
KEYPOINT_2D = slice(26, 28)
KEYPOINT_2D_VALID = 28
U0_RELIABILITY = 29
TORSO_POSITION = slice(30, 33)
TORSO_POSITION_VALID = 33
WRIST_POSITION = slice(34, 37)
WRIST_POSITION_VALID = 37
PALM_NORMAL = slice(38, 41)
PALM_VALID = 41
TIME_DELTA = 42
REPROJECTION_RESIDUAL_2D = slice(43, 45)
REPROJECTION_RESIDUAL_SCALE = 10.0
TOKEN_FEATURE_DIM = 43
TOKEN_FEATURE_DIM_WITH_REPROJECTION = 45


def _masked_coordinates(value: np.ndarray, valid: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(value).float()
    mask = torch.from_numpy(valid).bool()
    return torch.where(mask[..., None], tensor, torch.zeros_like(tensor))


def _keypoints_in_model_coordinates(clip, window: slice) -> np.ndarray:
    """Standardize 2D cache coordinates to normalized image [-1, 1]."""
    metadata = json.loads(clip.metadata_json)
    policy = metadata.get("coordinate_policy", {})
    coordinate_system = (
        policy.get("keypoints_2d", "") if isinstance(policy, dict) else ""
    )
    value = clip.keypoints_2d[window]
    # Early How2Sign-v1 caches predate the explicit policy field, but their
    # extractor contract is unambiguously normalized image [0, 1].
    legacy_how2sign = (
        not coordinate_system and str(metadata.get("dataset", "")).lower() == "how2sign"
    )
    if coordinate_system == "normalized_image_0_to_1" or legacy_how2sign:
        return value * 2.0 - 1.0
    return value


def features_from_clip(
    clip,
    window: slice | None = None,
    input_dim: int = TOKEN_FEATURE_DIM,
    reprojection_residual_scale: float = REPROJECTION_RESIDUAL_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip.validate()
    window = window or slice(0, len(clip.frame_names))
    init_aa = torch.from_numpy(clip.init_axis_angle[window]).float()
    init_matrix = axis_angle_to_matrix(init_aa)
    rot6d = matrix_to_rotation_6d(init_matrix)
    velocity = torch.zeros_like(rot6d)
    acceleration = torch.zeros_like(rot6d)
    if len(rot6d) > 1:
        velocity[1:] = rot6d[1:] - rot6d[:-1]
    if len(rot6d) > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]

    observations = torch.from_numpy(clip.observation_features[window]).float()
    keypoint_valid = torch.from_numpy(clip.keypoint_valid[window]).bool()
    keypoints = _masked_coordinates(
        _keypoints_in_model_coordinates(clip, window), clip.keypoint_valid[window]
    )
    reliability = torch.from_numpy(clip.u0_reliability[window]).float()[..., None]
    torso_valid = torch.from_numpy(clip.torso_position_valid[window]).bool()
    torso = _masked_coordinates(
        clip.torso_positions[window], clip.torso_position_valid[window]
    )
    wrist_valid = torch.from_numpy(clip.wrist_local_valid[window]).bool()
    wrist = _masked_coordinates(
        clip.wrist_local_positions[window], clip.wrist_local_valid[window]
    )

    palm = torch.zeros(len(init_aa), NUM_JOINTS, 3, dtype=torch.float32)
    palm_valid = torch.zeros(len(init_aa), NUM_JOINTS, dtype=torch.bool)
    palm_source = torch.from_numpy(clip.palm_normals[window]).float()
    palm_source_valid = torch.from_numpy(clip.palm_valid[window]).bool()
    palm[:, 21:36] = palm_source[:, 0, None]
    palm[:, 36:51] = palm_source[:, 1, None]
    palm_valid[:, 21:36] = palm_source_valid[:, 0, None]
    palm_valid[:, 36:51] = palm_source_valid[:, 1, None]
    palm = torch.where(palm_valid[..., None], palm, torch.zeros_like(palm))

    timestamps = torch.from_numpy(clip.timestamps[window]).float()
    time_delta = torch.zeros(len(timestamps), dtype=torch.float32)
    if len(timestamps) > 1:
        time_delta[1:] = timestamps[1:] - timestamps[:-1]
        nominal = 1.0 / clip.fps if clip.fps > 0 else time_delta[1:].median()
        time_delta = time_delta / torch.as_tensor(nominal).clamp_min(1e-6)
    time_delta = time_delta[:, None, None].expand(-1, NUM_JOINTS, 1)
    reprojection_residual = _masked_coordinates(
        clip.reprojection_residual_2d[window], clip.keypoint_valid[window]
    )

    components = [
        rot6d,
        velocity,
        acceleration,
        observations,
        keypoints,
        keypoint_valid[..., None].float(),
        reliability,
        torso,
        torso_valid[..., None].float(),
        wrist,
        wrist_valid[..., None].float(),
        palm,
        palm_valid[..., None].float(),
        time_delta,
    ]
    if input_dim == TOKEN_FEATURE_DIM_WITH_REPROJECTION:
        components.append(reprojection_residual * reprojection_residual_scale)
    elif input_dim != TOKEN_FEATURE_DIM:
        raise ValueError(
            f"Unsupported token feature dimension {input_dim}; expected "
            f"{TOKEN_FEATURE_DIM} or {TOKEN_FEATURE_DIM_WITH_REPROJECTION}"
        )
    features = torch.cat(components, dim=-1)
    if features.shape[-1] != input_dim:
        raise AssertionError(features.shape)
    return features, init_matrix


def collate_sequences(items: list[dict]) -> dict:
    tensor_keys = (
        "features",
        "initial_matrix",
        "target_matrix",
        "target_rotation_valid",
        "initial_joint_position",
        "target_joint_position",
        "target_joint_valid",
        "target_palm_normal",
        "target_palm_valid",
        "betas",
        "global_orient",
        "transl",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
        "joint_valid",
        "frame_valid",
        "refine_mask",
        "length",
    )
    batch = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    batch["clip_id"] = [item["clip_id"] for item in items]
    batch["frame_names"] = [item["frame_names"] for item in items]
    return batch


class SequenceCacheDataset(Dataset):
    def __init__(
        self,
        cache_glob: str,
        max_frames: int = 64,
        training: bool = False,
        identity_target: bool = False,
        seed: int = 42,
        input_dim: int = TOKEN_FEATURE_DIM,
        reprojection_residual_scale: float = REPROJECTION_RESIDUAL_SCALE,
    ) -> None:
        manifest_path = Path(cache_glob)
        if manifest_path.suffix == ".json" and manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            entries = manifest.get("clips", manifest)
            if not isinstance(entries, list):
                raise ValueError(f"Invalid split manifest: {manifest_path}")
            self.paths = [
                (manifest_path.parent / entry).resolve()
                if not Path(entry).is_absolute()
                else Path(entry)
                for entry in entries
            ]
        else:
            self.paths = [Path(path) for path in sorted(glob.glob(cache_glob))]
        if not self.paths:
            raise ValueError(f"No cache files match: {cache_glob}")
        self.max_frames = max_frames
        self.training = training
        self.identity_target = identity_target
        self.input_dim = input_dim
        self.reprojection_residual_scale = float(reprojection_residual_scale)
        if self.reprojection_residual_scale <= 0:
            raise ValueError("reprojection_residual_scale must be positive")
        self.rng = np.random.default_rng(seed)
        self.lengths = [len(load_cache_clip(path).frame_names) for path in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def _window(self, length: int) -> slice:
        if length <= self.max_frames:
            return slice(0, length)
        if self.training:
            start = int(self.rng.integers(0, length - self.max_frames + 1))
        else:
            start = (length - self.max_frames) // 2
        return slice(start, start + self.max_frames)

    @staticmethod
    def _pad_time(value: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return value
        padding = [0, 0] * (value.ndim - 1) + [0, pad]
        return torch.nn.functional.pad(value, tuple(padding))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | list[str]]:
        clip = load_cache_clip(self.paths[index])
        window = self._window(len(clip.frame_names))
        features, init_matrix = features_from_clip(
            clip,
            window,
            input_dim=self.input_dim,
            reprojection_residual_scale=self.reprojection_residual_scale,
        )
        init_aa = torch.from_numpy(clip.init_axis_angle[window]).float()

        target_np = clip.target_axis_angle
        if target_np is None:
            if not self.identity_target:
                raise ValueError(
                    f"{self.paths[index]} has no target. Use identity_target only for smoke tests."
                )
            target_aa = init_aa.clone()
            target_rotation_valid = torch.ones_like(init_aa[..., 0], dtype=torch.bool)
        else:
            target_aa = torch.from_numpy(target_np[window]).float()
            target_rotation_valid = torch.from_numpy(
                clip.target_rotation_valid[window]
            ).bool()
        target_matrix = axis_angle_to_matrix(target_aa)

        initial_joint = torch.from_numpy(clip.torso_positions[window]).float()
        wrist_joint = torch.from_numpy(clip.wrist_local_positions[window]).float()
        wrist_joint_valid = torch.from_numpy(clip.wrist_local_valid[window]).bool()
        initial_joint = torch.where(
            wrist_joint_valid[..., None], wrist_joint, initial_joint
        )
        if clip.target_joint_positions is None:
            target_joint = initial_joint.clone()
            target_joint_valid = torch.zeros_like(
                initial_joint[..., 0], dtype=torch.bool
            )
        else:
            target_joint = torch.from_numpy(clip.target_joint_positions[window]).float()
            target_joint_valid = torch.from_numpy(
                clip.target_joint_valid[window]
            ).bool()
        target_palm = torch.stack(
            (
                palm_normal(target_joint[:, 21:36], "left"),
                palm_normal(target_joint[:, 36:51], "right"),
            ),
            dim=1,
        )
        left_mcp = torch.as_tensor([21 + index for index in MCP_INDICES])
        right_mcp = torch.as_tensor([36 + index for index in MCP_INDICES])
        target_palm_valid = torch.stack(
            (
                target_joint_valid[:, left_mcp].all(dim=-1),
                target_joint_valid[:, right_mcp].all(dim=-1),
            ),
            dim=-1,
        )

        length = len(init_aa)
        pad = self.max_frames - length
        features = self._pad_time(features, pad)
        init_matrix = self._pad_time(init_matrix, pad)
        target_matrix = self._pad_time(target_matrix, pad)
        target_rotation_valid = self._pad_time(target_rotation_valid, pad)
        initial_joint = self._pad_time(initial_joint, pad)
        target_joint = self._pad_time(target_joint, pad)
        target_joint_valid = self._pad_time(target_joint_valid, pad)
        target_palm = self._pad_time(target_palm, pad)
        target_palm_valid = self._pad_time(target_palm_valid, pad)
        joint_valid = self._pad_time(
            torch.from_numpy(clip.keypoint_valid[window]).bool(), pad
        )
        global_orient = self._pad_time(
            torch.from_numpy(clip.global_orient[window]).float(), pad
        )
        transl = self._pad_time(torch.from_numpy(clip.transl[window]).float(), pad)
        jaw_pose = self._pad_time(torch.from_numpy(clip.jaw_pose[window]).float(), pad)
        leye_pose = self._pad_time(
            torch.from_numpy(clip.leye_pose[window]).float(), pad
        )
        reye_pose = self._pad_time(
            torch.from_numpy(clip.reye_pose[window]).float(), pad
        )
        expression = self._pad_time(
            torch.from_numpy(clip.expression[window]).float(), pad
        )
        frame_valid = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_valid[:length] = True
        return {
            "clip_id": clip.clip_id,
            "frame_names": clip.frame_names[window].astype(str).tolist(),
            "features": features,
            "initial_matrix": init_matrix,
            "target_matrix": target_matrix,
            "target_rotation_valid": target_rotation_valid,
            "initial_joint_position": initial_joint,
            "target_joint_position": target_joint,
            "target_joint_valid": target_joint_valid,
            "target_palm_normal": target_palm,
            "target_palm_valid": target_palm_valid,
            "betas": torch.from_numpy(clip.betas).float(),
            "global_orient": global_orient,
            "transl": transl,
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "expression": expression,
            "joint_valid": joint_valid,
            "frame_valid": frame_valid,
            "refine_mask": torch.from_numpy(clip.refine_mask).bool(),
            "length": torch.tensor(length, dtype=torch.long),
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar clip lengths while retaining deterministic epoch shuffling."""

    def __init__(
        self,
        dataset: SequenceCacheDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.cursor = 0

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = sorted(range(len(self.dataset)), key=self.dataset.lengths.__getitem__)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
            for batch in batches:
                permutation = torch.randperm(len(batch), generator=generator).tolist()
                batch[:] = [batch[index] for index in permutation]
        while self.cursor < len(batches):
            batch = batches[self.cursor]
            self.cursor += 1
            yield batch
        self.epoch += 1
        self.cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
