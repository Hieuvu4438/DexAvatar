from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import AdapterOutput, ClipContext


class NpzEstimatorAdapter:
    """Read precomputed estimator outputs without importing estimator code."""

    def __init__(self, source_name: str, root: str | Path) -> None:
        self.source_name = source_name
        self.root = Path(root)

    def validate_assets(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def infer(self, clip: ClipContext) -> AdapterOutput:
        path = self.root / f"{clip.clip_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        tensors = {
            key: torch.from_numpy(data[key])
            for key in ("joints_3d", "rotations", "keypoints_2d", "features")
            if key in data
        }
        masks = {
            key: torch.from_numpy(data[key]).bool()
            for key in ("valid_3d", "valid_rot", "valid_2d")
            if key in data
        }
        output = AdapterOutput(
            source_name=self.source_name,
            tensors=tensors,
            masks=masks,
            metadata={
                "coordinate_convention": "opencv_x_right_y_down_z_forward",
                "length_unit": "meter",
            },
            diagnostics={"input_path": str(path)},
        )
        output.validate_length(len(clip.frame_ids))
        return output

    def canonicalize(self, raw: AdapterOutput) -> AdapterOutput:
        raw.validate_length(next(iter(raw.tensors.values())).shape[0])
        return raw
