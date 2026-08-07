"""Materialize relation/contact sidecars from immutable Phase 2 clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase2_refiner.geometry.smplx_decode import decode_smplx_sequence
from phase2_refiner.render import create_smplx_model
from phase3_posterior.data.cache_schema import RelationSidecar, save_relation_sidecar
from phase3_posterior.geometry.contact import (
    contact_hysteresis,
    contact_persistence_target,
)
from phase3_posterior.geometry.relation_anchors import (
    CONTACT_PROXY_RADII_M,
    build_edge_features,
    build_relation_geometry,
    build_relation_nodes,
    default_edge_index,
)
from phase3_posterior.provenance import sha256_file


_STATE_TO_SMPLX_JOINT = (
    *range(1, 22),
    25,
    26,
    67,
    28,
    29,
    68,
    31,
    32,
    70,
    34,
    35,
    69,
    37,
    38,
    66,
    40,
    41,
    72,
    43,
    44,
    73,
    46,
    47,
    75,
    49,
    50,
    74,
    52,
    53,
    71,
)

_INTERHAND_TO_MANO = {
    "right": np.asarray([7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3, 2, 1]),
    "left": np.asarray([28, 27, 26, 32, 31, 30, 40, 39, 38, 36, 35, 34, 24, 23, 22]),
}
_INTERHAND_WRIST = {"right": 20, "left": 41}
_TOKEN_OFFSET = {"left": 21, "right": 36}


class InterHandJointProvider:
    """Lazy access to official two-hand world coordinates for relation labels."""

    def __init__(self) -> None:
        self._payloads: dict[Path, dict] = {}

    def geometry(self, clip, metadata: dict, device: torch.device):
        root = Path(metadata["annotation_root"])
        split = str(metadata["official_split"])
        path = root / split / f"InterHand2.6M_{split}_joint_3d.json"
        if path not in self._payloads:
            self._payloads[path] = json.loads(path.read_text(encoding="utf-8"))
        capture = str(metadata["capture"])
        frames = self._payloads[path][capture]
        length = len(clip.frame_numbers)
        torso = np.zeros((length, 51, 3), dtype=np.float32)
        torso_valid = np.zeros((length, 51), dtype=bool)
        wrist_local = np.zeros_like(torso)
        wrist_valid = np.zeros_like(torso_valid)
        transforms = np.broadcast_to(
            np.eye(4, dtype=np.float32), (length, 2, 4, 4)
        ).copy()
        for index, frame_number in enumerate(clip.frame_numbers):
            world = np.asarray(
                frames[str(int(frame_number))]["world_coord"], np.float32
            )
            world /= 1000.0
            wrists = []
            for side in ("left", "right"):
                wrists.append(world[_INTERHAND_WRIST[side]])
            origin = np.mean(wrists, axis=0)
            for hand_index, side in enumerate(("left", "right")):
                offset = _TOKEN_OFFSET[side]
                wrist = world[_INTERHAND_WRIST[side]]
                hand = world[_INTERHAND_TO_MANO[side]] - wrist
                wrist_local[index, offset : offset + 15] = hand
                wrist_valid[index, offset : offset + 15] = True
                torso[index, 19 + hand_index] = wrist - origin
                torso_valid[index, 19 + hand_index] = True
                transforms[index, hand_index, :3, 3] = wrist - origin
        return tuple(
            torch.as_tensor(value, device=device)
            for value in (torso, torso_valid, wrist_local, wrist_valid, transforms)
        )


@torch.inference_mode()
def _decode_geometry(clip, pose: np.ndarray, model, device: torch.device):
    """Decode state joints and express them in a pelvis-centred torso frame."""
    matrices = axis_angle_to_matrix(torch.as_tensor(pose, device=device).float())[None]
    _, joints = decode_smplx_sequence(
        model,
        matrices,
        torch.as_tensor(clip.betas, device=device).float()[None],
        torch.as_tensor(clip.global_orient, device=device).float()[None],
        torch.as_tensor(clip.transl, device=device).float()[None],
        jaw_pose=torch.as_tensor(clip.jaw_pose, device=device).float()[None],
        leye_pose=torch.as_tensor(clip.leye_pose, device=device).float()[None],
        reye_pose=torch.as_tensor(clip.reye_pose, device=device).float()[None],
        expression=torch.as_tensor(clip.expression, device=device).float()[None],
    )
    selected = joints[0, :, list(_STATE_TO_SMPLX_JOINT)]
    pelvis = joints[0, :, :1]
    global_rotation = axis_angle_to_matrix(
        torch.as_tensor(clip.global_orient, device=device).float()
    )
    torso = (selected - pelvis) @ global_rotation
    valid = torch.isfinite(torso).all(dim=-1)
    torso = torch.nan_to_num(torso)
    wrist_local = torch.zeros_like(torso)
    wrist_local[:, 21:36] = torso[:, 21:36] - torso[:, 19:20]
    wrist_local[:, 36:51] = torso[:, 36:51] - torso[:, 20:21]
    wrist_valid = torch.zeros_like(valid)
    wrist_valid[:, 21:36] = valid[:, 21:36] & valid[:, 19:20]
    wrist_valid[:, 36:51] = valid[:, 36:51] & valid[:, 20:21]
    transform = (
        torch.eye(4, dtype=torso.dtype, device=device)
        .expand(torso.shape[0], 2, 4, 4)
        .clone()
    )
    transform[:, 0, :3, 3] = torso[:, 19]
    transform[:, 1, :3, 3] = torso[:, 20]
    return torso, valid, wrist_local, wrist_valid, transform


def _cached_geometry(clip):
    return (
        torch.from_numpy(clip.torso_positions).float(),
        torch.from_numpy(clip.torso_position_valid).bool(),
        torch.from_numpy(clip.wrist_local_positions).float(),
        torch.from_numpy(clip.wrist_local_valid).bool(),
        torch.from_numpy(clip.wrist_to_torso).float(),
    )


def build_sidecar(
    clip_path: str | Path,
    model=None,
    device: str | torch.device = "cpu",
    interhand_provider: InterHandJointProvider | None = None,
) -> RelationSidecar:
    source = Path(clip_path).resolve()
    clip = load_cache_clip(source)
    device = torch.device(device)
    source_metadata = json.loads(clip.metadata_json)
    if source_metadata.get("dataset") == "InterHand2.6M":
        if interhand_provider is None:
            interhand_provider = InterHandJointProvider()
        initializer_fields = interhand_provider.geometry(clip, source_metadata, device)
        target_fields = initializer_fields
        geometry_provider = "interhand_official_joint_3d_v1"
        target_provider = geometry_provider
    elif model is None:
        initializer_fields = _cached_geometry(clip)
        target_fields = initializer_fields
        geometry_provider = "phase2_cache_geometry"
        target_provider = geometry_provider
    else:
        # These Phase 2 caches may not carry valid 3D relation anchors, so the
        # immutable initializer pose and independent target pose are decoded
        # explicitly through the same frozen SMPL-X geometry contract.
        initializer_fields = _decode_geometry(clip, clip.init_axis_angle, model, device)
        if clip.target_axis_angle is None:
            raise ValueError(f"Missing target_axis_angle for {clip.clip_id}")
        target_fields = (
            initializer_fields
            if np.array_equal(clip.init_axis_angle, clip.target_axis_angle)
            else _decode_geometry(clip, clip.target_axis_angle, model, device)
        )
        geometry_provider = "smplx_decoded_initializer_v2"
        target_provider = (
            geometry_provider
            if target_fields is initializer_fields
            else "smplx_decoded_independent_target_v2"
        )
    geometry = build_relation_geometry(*initializer_fields)
    target_nodes, target_node_valid = build_relation_nodes(*target_fields)
    target_features, target_edge_valid = build_edge_features(
        target_nodes, target_node_valid, default_edge_index(target_nodes.device)
    )
    source_node, target_node = geometry.edge_index
    radii = CONTACT_PROXY_RADII_M.to(target_features.device)
    distance = target_features[..., 3] - (radii[source_node] + radii[target_node])
    contact_valid = target_edge_valid.clone()
    # The final four edges encode own-chain kinematics, not candidate contact.
    contact_valid[..., -4:] = False
    contact = contact_hysteresis(distance, contact_valid)
    relative_speed = torch.linalg.vector_norm(target_features[..., 4:7], dim=-1)
    persistence = contact_persistence_target(contact) & (relative_speed < 0.15)
    depth = target_features[..., 10]
    depth_target = torch.ones_like(depth, dtype=torch.long)
    depth_target[depth < -0.005] = 0
    depth_target[depth > 0.005] = 2
    metadata = {
        "source_clip": str(source),
        "source_sha256": sha256_file(source),
        "contact_onset_m": 0.012,
        "contact_release_m": 0.020,
        "contact_semantics": "geometry-derived candidate; not phonology",
        "contact_distance": "decoded centre distance minus fixed anatomical proxy radii",
        "contact_radii_m": CONTACT_PROXY_RADII_M.tolist(),
        "persistence_speed_threshold_mps": 0.15,
        "input_geometry_provider": geometry_provider,
        "target_geometry_provider": target_provider,
        "coordinate_frame": "pelvis-centred, global-orientation-cancelled SMPL-X",
    }
    return RelationSidecar(
        clip_id=clip.clip_id,
        node_positions=geometry.nodes.cpu().numpy().astype(np.float32),
        node_valid=geometry.node_valid.cpu().numpy(),
        edge_index=geometry.edge_index.cpu().numpy(),
        edge_features=geometry.edge_features.cpu().numpy().astype(np.float32),
        edge_valid=geometry.edge_valid.cpu().numpy(),
        contact_target=contact.cpu().numpy(),
        contact_valid=contact_valid.cpu().numpy(),
        persistence_target=persistence.cpu().numpy(),
        depth_target=depth_target.cpu().numpy().astype(np.int64),
        target_edge_features=target_features.cpu().numpy().astype(np.float32),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-folder")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = None
    if args.model_folder:
        model = create_smplx_model(args.model_folder, torch.device(args.device))
    save_relation_sidecar(
        args.output, build_sidecar(args.clip, model=model, device=args.device)
    )


if __name__ == "__main__":
    main()
