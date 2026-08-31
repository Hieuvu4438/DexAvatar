from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dcg_sign4d.geometry.smplx_adapter import SMPLXAdapter
from dcg_sign4d.initialization.artifact import (
    load_initialization_artifact,
    save_initialization_artifact,
)
from dcg_sign4d.initialization.camera import CameraTrajectory
from dcg_sign4d.initialization.dexavatar_adapter import DexAvatarPklInitializer
from dcg_sign4d.utils.hashing import file_sha256


def _to_device(state, device):
    return replace(
        state,
        **{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name in state.__dataclass_fields__
            if (value := getattr(state, name)) is not None
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert frozen DexAvatar PKLs to safe replay")
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--dexavatar-commit", required=True)
    parser.add_argument("--dexavatar-config-sha256", required=True)
    parser.add_argument("--dexavatar-checkpoint-sha256", required=True)
    parser.add_argument("--image-width", required=True, type=int)
    parser.add_argument("--image-height", required=True, type=int)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-local-assets", action="store_true")
    args = parser.parse_args()
    if not args.trusted_local_assets:
        raise PermissionError("conversion requires --trusted-local-assets")
    registry_path = Path(args.source_registry)
    if file_sha256(registry_path) != args.expected_registry_sha256:
        raise ValueError("source registry hash mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["status"] != "frozen_before_pickle_conversion":
        raise ValueError("source registry was not frozen before conversion")
    if file_sha256(registry["manifest"]) != registry["manifest_sha256"]:
        raise ValueError("source manifest changed after registry freeze")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable initialization output exists: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".conversion_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    device = torch.device(args.device)
    adapter = SMPLXAdapter(
        args.model,
        expected_sha256=args.expected_model_sha256,
        trusted_model=True,
    ).to(device)
    initializer = DexAvatarPklInitializer(args.fps)
    summary = []
    for clip_index, clip in enumerate(registry["clips"], start=1):
        clip_id = clip["clip_id"]
        print(f"[{clip_index:02d}/{len(registry['clips']):02d}] {clip_id}", flush=True)
        paths = [Path(row["path"]) for row in clip["files"]]
        parent = paths[0].parent
        if any(path.parent != parent for path in paths):
            raise ValueError(f"mixed result directories for {clip_id}")
        expected_hashes = {
            path.name: row["sha256"] for path, row in zip(paths, clip["files"], strict=True)
        }
        state, metadata = initializer.reconstruct_from_directory(
            parent,
            expected_hashes=expected_hashes,
            include_names=set(expected_hashes),
            trusted=True,
        )
        expected_frame_ids = [row["frame_id"] for row in clip["files"]]
        if metadata["frame_ids"] != expected_frame_ids:
            raise ValueError(f"frame mismatch for {clip_id}")
        clip_root = output / clip_id
        camera_intrinsics = torch.tensor(metadata.pop("camera_intrinsics"), dtype=torch.float32)
        # SMPLXAdapter exposes vertices in DexAvatar's X-180 world convention.
        # Apply the inverse (equal) X-180 camera transform so pinhole depth is
        # positive and exactly matches the coordinates used during fitting.
        world_to_camera = (
            torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))[None, None]
            .expand(1, len(expected_frame_ids), 4, 4)
            .clone()
        )
        image_size = (
            torch.tensor([args.image_width, args.image_height], dtype=torch.float32)[None, None]
            .expand(1, len(expected_frame_ids), 2)
            .clone()
        )
        camera = CameraTrajectory(
            intrinsics=camera_intrinsics,
            world_to_camera=world_to_camera,
            image_size_wh=image_size,
            valid_mask=state.valid_mask,
            coordinate_convention="dexavatar_camera_x_180",
        ).validate()
        metadata.update(
            {
                "source_registry_sha256": args.expected_registry_sha256,
                "smplx_model_sha256": args.expected_model_sha256,
                "development_only": True,
                "signer_id_status": "unknown",
                "clip_id": clip_id,
                "dexavatar_commit": args.dexavatar_commit,
                "config_sha256": args.dexavatar_config_sha256,
                "checkpoint_sha256": args.dexavatar_checkpoint_sha256,
                "runtime": {"mode": "existing_frozen_pkl_conversion"},
            }
        )
        device_state = _to_device(state, device)
        with torch.inference_mode():
            forward = adapter(device_state)
        save_initialization_artifact(
            clip_root,
            state,
            camera,
            metadata=metadata,
            source_hashes={
                **expected_hashes,
                "source_registry": args.expected_registry_sha256,
                "smplx_model": args.expected_model_sha256,
            },
            smplx_forward={
                "vertices": forward.vertices.cpu().numpy(),
                "joints": forward.joints.cpu().numpy(),
                "frame_ids": np.asarray(expected_frame_ids, dtype=np.int64),
            },
        )
        replayed, replay_camera, _ = load_initialization_artifact(clip_root)
        if not torch.equal(replay_camera.intrinsics, camera.intrinsics):
            raise ValueError(f"camera replay mismatch for {clip_id}")
        with torch.inference_mode():
            replay_forward = adapter(_to_device(replayed, device)).vertices.cpu()
        max_replay_error = float((replay_forward - forward.vertices.cpu()).abs().max())
        if max_replay_error > 1e-7:
            raise ValueError(f"trajectory replay mismatch for {clip_id}: {max_replay_error}")
        summary.append(
            {
                "clip_id": clip_id,
                "frames": len(expected_frame_ids),
                "max_replay_error_m": max_replay_error,
                "max_beta_deviation": metadata["max_beta_deviation"],
            }
        )
    report = {
        "schema_version": "1.0",
        "development_only": True,
        "clips": len(summary),
        "frames": sum(row["frames"] for row in summary),
        "source_registry_sha256": args.expected_registry_sha256,
        "smplx_model_sha256": args.expected_model_sha256,
        "max_replay_error_m": max(row["max_replay_error_m"] for row in summary),
        "per_clip": summary,
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(incomplete, output / "CONVERSION_COMPLETE")
    print(json.dumps({key: value for key, value in report.items() if key != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
