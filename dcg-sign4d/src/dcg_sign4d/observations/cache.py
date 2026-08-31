"""Immutable, hash-addressed observation cache."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

from .schema import ObservationBatch


class ObservationCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def identity(
        *,
        video_hash: str,
        extractor: dict[str, Any],
        preprocessing: dict[str, Any],
        calibration_hash: str,
    ) -> str:
        return canonical_hash(
            {
                "video_hash": video_hash,
                "extractor": extractor,
                "preprocessing": preprocessing,
                "calibration_hash": calibration_hash,
            }
        )

    def save(self, cache_id: str, observations: ObservationBatch) -> Path:
        observations.validate()
        destination = self.root / cache_id
        if destination.exists():
            raise FileExistsError(f"immutable cache already exists: {destination}")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_id}.", dir=self.root))
        try:
            arrays: dict[str, np.ndarray] = {}
            for name in observations.__dataclass_fields__:
                value = getattr(observations, name)
                if isinstance(value, torch.Tensor):
                    arrays[name] = value.detach().cpu().numpy()
            data_path = temporary / "observations.npz"
            metadata_path = temporary / "metadata.json"
            np.savez_compressed(data_path, **arrays)
            metadata_path.write_text(
                json.dumps(list(observations.metadata), sort_keys=True, indent=2),
                encoding="utf-8",
            )
            identity = {
                "schema_version": "dcg_observation_cache_v1",
                "cache_id": cache_id,
                "observations_sha256": file_sha256(data_path),
                "metadata_sha256": file_sha256(metadata_path),
            }
            identity["identity_sha256"] = canonical_hash(identity)
            (temporary / "identity.json").write_text(
                json.dumps(identity, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            (temporary / "CACHE_COMPLETE").write_text("complete\n", encoding="utf-8")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def load(self, cache_id: str) -> ObservationBatch:
        source = self.root / cache_id
        if not (source / "CACHE_COMPLETE").is_file():
            raise ValueError("observation cache has no completion marker")
        data_path = source / "observations.npz"
        metadata_path = source / "metadata.json"
        identity = json.loads((source / "identity.json").read_text(encoding="utf-8"))
        stored = identity.pop("identity_sha256", None)
        if stored != canonical_hash(identity):
            raise ValueError("observation cache identity mismatch")
        if identity.get("cache_id") != cache_id:
            raise ValueError("observation cache directory/identity mismatch")
        if file_sha256(data_path) != identity.get("observations_sha256"):
            raise ValueError("observation cache tensor hash mismatch")
        if file_sha256(metadata_path) != identity.get("metadata_sha256"):
            raise ValueError("observation cache metadata hash mismatch")
        arrays = np.load(data_path, allow_pickle=False)
        metadata = tuple(json.loads(metadata_path.read_text(encoding="utf-8")))

        def tensor(name: str, *, boolean: bool = False) -> torch.Tensor | None:
            if name not in arrays:
                return None
            result = torch.from_numpy(arrays[name])
            return result.bool() if boolean else result

        return ObservationBatch(
            keypoints_2d=tensor("keypoints_2d"),
            keypoint_reliability=tensor("keypoint_reliability"),
            keypoint_valid=tensor("keypoint_valid", boolean=True),
            frame_valid=tensor("frame_valid", boolean=True),
            part_masks=tensor("part_masks"),
            mask_reliability=tensor("mask_reliability"),
            tracks_2d=tensor("tracks_2d"),
            track_reliability=tensor("track_reliability"),
            depth_order=tensor("depth_order"),
            depth_reliability=tensor("depth_reliability"),
            metadata=metadata,
        ).validate()
