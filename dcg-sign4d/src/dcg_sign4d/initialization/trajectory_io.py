"""Replayable trajectory artifact without pickle execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.utils.hashing import file_sha256


def save_trajectory(state: TrajectoryState, root: str | Path, metadata: dict[str, Any]) -> Path:
    state.validate()
    destination = Path(root)
    if destination.exists():
        raise FileExistsError(f"immutable trajectory artifact exists: {destination}")
    destination.mkdir(parents=True)
    arrays = {
        name: value.detach().cpu().numpy()
        for name in state.__dataclass_fields__
        if isinstance((value := getattr(state, name)), torch.Tensor)
    }
    target = destination / "trajectory.npz"
    np.savez_compressed(target, **arrays)
    identity = dict(metadata)
    identity["trajectory_sha256"] = file_sha256(target)
    identity["schema_version"] = "trajectory_state_v1"
    (destination / "metadata.json").write_text(
        json.dumps(identity, sort_keys=True, indent=2), encoding="utf-8"
    )
    return destination


def load_trajectory(root: str | Path) -> tuple[TrajectoryState, dict[str, Any]]:
    source = Path(root)
    target = source / "trajectory.npz"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if file_sha256(target) != metadata["trajectory_sha256"]:
        raise ValueError(f"trajectory hash mismatch: {target}")
    arrays = np.load(target, allow_pickle=False)

    def tensor(name: str, *, boolean: bool = False) -> torch.Tensor | None:
        if name not in arrays:
            return None
        value = torch.from_numpy(arrays[name])
        return value.bool() if boolean else value

    return (
        TrajectoryState(
            root_rot6d=tensor("root_rot6d"),
            root_translation=tensor("root_translation"),
            root_velocity=tensor("root_velocity"),
            body_rot6d=tensor("body_rot6d"),
            left_hand_rot6d=tensor("left_hand_rot6d"),
            right_hand_rot6d=tensor("right_hand_rot6d"),
            face_state=tensor("face_state"),
            beta=tensor("beta"),
            valid_mask=tensor("valid_mask", boolean=True),
        ).validate(),
        metadata,
    )
