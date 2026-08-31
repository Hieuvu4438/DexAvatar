"""Freeze the DexAvatar COCO-WholeBody to SMPL-X projection map."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-definitions", required=True)
    parser.add_argument("--runtime-joint-count", type=int, default=127)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.joint_definitions)
    namespace = runpy.run_path(str(source))
    coco = namespace["COCO_WHOLEBODY_KEYPOINTS"]
    smplx = namespace["SMPLX_KEYPOINTS"]
    aliases = {"left_hand_root": "left_wrist", "right_hand_root": "right_wrist"}
    mapped_names = [aliases.get(name, name) for name in coco]
    missing = [name for name in mapped_names if name not in smplx]
    if missing:
        raise ValueError(f"COCO WholeBody names missing from SMPL-X topology: {missing}")
    indices = [smplx.index(name) for name in mapped_names]
    supported = [index < args.runtime_joint_count for index in indices]
    payload = {
        "schema_version": "dcg_camera_projection_v1",
        "scientific_status": "DEVELOPMENT",
        "coordinate_convention": "dexavatar_camera_x_180",
        "observation_topology": "COCO_WHOLEBODY_KEYPOINTS_133",
        "smplx_joint_topology": "DexAvatar_SMPLX_KEYPOINTS",
        "runtime_joint_count": args.runtime_joint_count,
        "keypoint_joint_indices": [
            index if valid else 0 for index, valid in zip(indices, supported, strict=True)
        ],
        "keypoint_supported_mask": supported,
        "aliases": aliases,
        "source_mapping_file": str(source.resolve()),
        "source_mapping_sha256": file_sha256(source),
    }
    payload["asset_identity_sha256"] = canonical_hash(payload)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable camera projection asset exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", "utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output.resolve()), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
