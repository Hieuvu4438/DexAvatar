"""Phase 3 dataset view over immutable Phase 2 clips and relation sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import SequenceCacheDataset, collate_sequences
from phase3_posterior.data.cache_schema import load_index, load_relation_sidecar
from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    default_edge_index,
)
from phase3_posterior.geometry.state_adapter import matrices_to_state


class Phase3Dataset(Dataset):
    def __init__(
        self,
        index_path: str,
        max_frames: int = 64,
        training: bool = False,
        seed: int = 42,
        input_dim: int = 45,
        identity_target: bool = False,
    ) -> None:
        self.entries = load_index(index_path)
        # Construct a private read-only view while retaining the proven Phase 2 item loader.
        self.base = SequenceCacheDataset.__new__(SequenceCacheDataset)
        self.base.paths = [Path(entry.clip_path) for entry in self.entries]
        self.base.max_frames = max_frames
        self.base.training = training
        self.base.identity_target = identity_target
        self.base.input_dim = input_dim
        self.base.reprojection_residual_scale = 10.0
        self.base.rng = np.random.default_rng(seed)
        self.base.lengths = [
            len(load_cache_clip(path).frame_names) for path in self.base.paths
        ]
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        entry = self.entries[index]
        item["initial_state"] = matrices_to_state(item["initial_matrix"])
        item["target_state"] = matrices_to_state(item["target_matrix"])
        item["target_weight"] = torch.tensor(entry.target_weight, dtype=torch.float32)
        item["source"] = entry.source
        edges = default_edge_index()
        edge_count = edges.shape[1]
        edge_features = torch.zeros(self.max_frames, edge_count, EDGE_FEATURE_DIM)
        edge_valid = torch.zeros(self.max_frames, edge_count, dtype=torch.bool)
        contact_target = torch.zeros(self.max_frames, edge_count, dtype=torch.bool)
        contact_valid = torch.zeros_like(contact_target)
        persistence_target = torch.zeros_like(contact_target)
        depth_target = torch.ones(self.max_frames, edge_count, dtype=torch.long)
        if entry.relation_path:
            relation = load_relation_sidecar(entry.relation_path)
            clip = load_cache_clip(entry.clip_path)
            frame_lookup = {
                str(name): frame_index
                for frame_index, name in enumerate(clip.frame_names.astype(str))
            }
            start = frame_lookup[item["frame_names"][0]]
            length = min(int(item["length"]), len(relation.node_positions))
            if not torch.equal(torch.from_numpy(relation.edge_index), edges):
                raise ValueError(
                    f"Relation edge contract mismatch: {entry.relation_path}"
                )
            window = slice(start, start + length)
            edge_features[:length] = torch.from_numpy(relation.edge_features[window])
            edge_valid[:length] = torch.from_numpy(relation.edge_valid[window])
            contact_target[:length] = torch.from_numpy(relation.contact_target[window])
            contact_valid[:length] = torch.from_numpy(relation.contact_valid[window])
            persistence_target[:length] = torch.from_numpy(
                relation.persistence_target[window]
            )
            depth_target[:length] = torch.from_numpy(relation.depth_target[window])
        item.update(
            edge_index=edges,
            edge_features=edge_features,
            edge_valid=edge_valid,
            contact_target=contact_target,
            contact_valid=contact_valid,
            persistence_target=persistence_target,
            depth_target=depth_target,
        )
        return item


def collate_phase3(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_sequences(items)
    tensor_keys = (
        "initial_state",
        "target_state",
        "target_weight",
        "edge_index",
        "edge_features",
        "edge_valid",
        "contact_target",
        "contact_valid",
        "persistence_target",
        "depth_target",
    )
    batch.update(
        {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    )
    batch["source"] = [item["source"] for item in items]
    return batch
