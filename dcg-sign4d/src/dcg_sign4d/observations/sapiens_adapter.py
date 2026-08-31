"""Safe adapter for existing Sapiens WholeBody JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dcg_sign4d.observations.raw import RawKeypointBatch


def load_sapiens_clip(
    clip_id: str,
    frame_ids: list[int],
    *,
    fps: float,
    detector_root: str | Path,
) -> tuple[RawKeypointBatch, list[Path]]:
    if fps <= 0 or not frame_ids:
        raise ValueError("raw observation extraction requires frames and positive fps")
    root = Path(detector_root) / clip_id / "sapiens_1b"
    points: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    available: list[np.ndarray] = []
    source_paths: list[Path] = []
    joint_count = 133
    for frame_id in frame_ids:
        path = root / f"low_{frame_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        instances = payload.get("instance_info")
        if not isinstance(instances, list):
            raise ValueError(f"invalid Sapiens instance_info: {path}")
        if not instances:
            points.append(np.zeros((joint_count, 2), dtype=np.float32))
            scores.append(np.zeros(joint_count, dtype=np.float32))
            available.append(np.zeros(joint_count, dtype=bool))
        else:
            if len(instances) != 1:
                raise ValueError(f"ambiguous multi-person Sapiens output: {path}")
            point = np.asarray(instances[0].get("keypoints"), dtype=np.float32)
            score = np.asarray(instances[0].get("keypoint_scores"), dtype=np.float32)
            if point.shape != (joint_count, 2) or score.shape != (joint_count,):
                raise ValueError(f"unexpected Sapiens WholeBody topology: {path}")
            structurally_available = np.isfinite(point).all(axis=-1) & np.isfinite(score)
            points.append(np.nan_to_num(point))
            scores.append(np.nan_to_num(score))
            available.append(structurally_available)
        source_paths.append(path)
    frame_available = np.asarray([bool(mask.any()) for mask in available])
    batch = RawKeypointBatch(
        frame_ids=torch.tensor(frame_ids, dtype=torch.long),
        timestamps_sec=torch.tensor(frame_ids, dtype=torch.float64) / fps,
        keypoints_2d=torch.from_numpy(np.stack(points)),
        raw_score=torch.from_numpy(np.stack(scores)),
        keypoint_available=torch.from_numpy(np.stack(available)),
        frame_available=torch.from_numpy(frame_available),
    ).validate()
    return batch, source_paths
