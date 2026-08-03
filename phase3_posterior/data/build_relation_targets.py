"""Materialize relation/contact sidecars from immutable Phase 2 clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.data.cache_schema import load_cache_clip
from phase3_posterior.data.cache_schema import RelationSidecar, save_relation_sidecar
from phase3_posterior.geometry.contact import (
    contact_hysteresis,
    contact_persistence_target,
)
from phase3_posterior.geometry.relation_anchors import build_relation_geometry
from phase3_posterior.provenance import sha256_file


def build_sidecar(clip_path: str | Path) -> RelationSidecar:
    source = Path(clip_path).resolve()
    clip = load_cache_clip(source)
    geometry = build_relation_geometry(
        torch.from_numpy(clip.torso_positions).float(),
        torch.from_numpy(clip.torso_position_valid).bool(),
        torch.from_numpy(clip.wrist_local_positions).float(),
        torch.from_numpy(clip.wrist_local_valid).bool(),
        torch.from_numpy(clip.wrist_to_torso).float(),
    )
    distance = geometry.edge_features[..., 3]
    contact_valid = geometry.edge_valid.clone()
    # The final four edges encode own-chain kinematics, not candidate contact.
    contact_valid[..., -4:] = False
    contact = contact_hysteresis(distance, contact_valid)
    persistence = contact_persistence_target(contact)
    depth = geometry.edge_features[..., 10]
    depth_target = torch.ones_like(depth, dtype=torch.long)
    depth_target[depth < -0.005] = 0
    depth_target[depth > 0.005] = 2
    metadata = {
        "source_clip": str(source),
        "source_sha256": sha256_file(source),
        "contact_onset_m": 0.012,
        "contact_release_m": 0.020,
        "contact_semantics": "geometry-derived candidate; not phonology",
    }
    return RelationSidecar(
        clip_id=clip.clip_id,
        node_positions=geometry.nodes.numpy().astype(np.float32),
        node_valid=geometry.node_valid.numpy(),
        edge_index=geometry.edge_index.numpy(),
        edge_features=geometry.edge_features.numpy().astype(np.float32),
        edge_valid=geometry.edge_valid.numpy(),
        contact_target=contact.numpy(),
        contact_valid=contact_valid.numpy(),
        persistence_target=persistence.numpy(),
        depth_target=depth_target.numpy().astype(np.int64),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_relation_sidecar(args.output, build_sidecar(args.clip))


if __name__ == "__main__":
    main()
