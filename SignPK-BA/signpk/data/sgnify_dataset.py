from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from .frame_manifest import SignManifest
from .window_sampler import TemporalWindow, all_windows


class SGNifyClipDataset(Dataset):
    """Manifest-backed central clip with explicit temporal padding metadata."""

    def __init__(self, manifest: str | Path | SignManifest, window: int = 9, gap: int = 1):
        self.manifest = (
            manifest if isinstance(manifest, SignManifest) else SignManifest.load(manifest, validate_paths=True)
        )
        self.windows = all_windows(
            len(self.manifest.records), window, gap, self.manifest.boundary_padding
        )

    def __len__(self) -> int:
        return len(self.manifest.records)

    @staticmethod
    def _read(path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return torch.from_numpy(image.copy())

    def __getitem__(self, index: int) -> dict[str, object]:
        spec: TemporalWindow = self.windows[index]
        records = [self.manifest.records[item] for item in spec.indices]
        images = torch.stack([self._read(record.rgb_path) for record in records])
        return {
            "images_bgr": images,
            "window_indices": torch.tensor(spec.indices, dtype=torch.int64),
            "padding_mask": torch.tensor(spec.padded, dtype=torch.bool),
            "padding_ratio": torch.tensor(spec.padding_ratio, dtype=torch.float32),
            "timestamp_sec": torch.tensor([record.timestamp_sec for record in records]),
            "center_video_frame_id": self.manifest.records[index].video_frame_id,
            "center_gt_frame_id": self.manifest.records[index].gt_frame_id,
        }

