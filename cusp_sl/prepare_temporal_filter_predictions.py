"""Apply the fixed target-free SO(3) A2 control to materialized A1 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cusp_sl.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from cusp_sl.temporal_filter import centered_tangent_filter, changed_joint_support
from phase2_refiner.data.cache_schema import load_cache_clip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.radius < 1:
        raise ValueError("A2 radius must be at least one")
    (args.output / "clips").mkdir(parents=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_manifest_path = args.predictions / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("role") != "frozen_a1_frontend_predictions":
        raise ValueError("A2 requires frozen A1 frontend predictions")
    declared = {
        str(item["clip_id"]): str(item["prediction_sha256"])
        for item in source_manifest["clips"]
    }
    summaries = []
    total_frames = 0
    for entry in manifest["clips"]:
        relative = entry["cache"] if isinstance(entry, dict) else entry
        cache_path = Path(relative)
        if not cache_path.is_absolute():
            cache_path = args.manifest.parent / cache_path
        clip = load_cache_clip(cache_path)
        source = args.predictions / "clips" / f"{clip.clip_id}.npz"
        if clip.clip_id not in declared or sha256(source) != declared[clip.clip_id]:
            raise ValueError(f"A1 source prediction hash mismatch: {source}")
        with np.load(source, allow_pickle=False) as payload:
            source_id = str(payload["clip_id"].item())
            names = payload["frame_names"].astype(str)
            selected = payload["selected_axis_angle"].astype(np.float32)
        if source_id != clip.clip_id or not np.array_equal(
            names, clip.frame_names.astype(str)
        ):
            raise ValueError(f"Prediction/cache identity mismatch: {source}")
        rotations = axis_angle_to_matrix(torch.from_numpy(selected))
        base = axis_angle_to_matrix(torch.from_numpy(clip.init_axis_angle).float())
        mask = changed_joint_support(base, rotations)
        mask &= torch.from_numpy(clip.refine_mask).bool()
        filtered = centered_tangent_filter(rotations, mask, radius=args.radius)
        axis_angle = matrix_to_axis_angle(filtered).numpy().astype(np.float32)
        output = args.output / "clips" / f"{clip.clip_id}.npz"
        np.savez_compressed(
            output,
            clip_id=np.asarray(clip.clip_id),
            frame_names=clip.frame_names,
            selected_axis_angle=axis_angle,
            selected_index=np.asarray(1, dtype=np.int64),
            candidate_valid=np.asarray([True, True]),
        )
        total_frames += len(clip.frame_names)
        summaries.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(clip.frame_names),
                "source_prediction_sha256": sha256(source),
                "filtered_prediction_sha256": sha256(output),
                "filtered_joint_indices": torch.where(mask)[0].tolist(),
            }
        )
        print(f"[a2] {clip.clip_id}: {len(clip.frame_names)} frames")

    if set(declared) != {item["clip_id"] for item in summaries}:
        raise ValueError("A1 prediction/source-manifest clip sets differ")

    expected = manifest.get("expected_frames")
    if expected is not None and total_frames != int(expected):
        raise ValueError(f"Filtered {total_frames} != expected {expected}")
    report = {
        "variant": "a2_simple_temporal_filter",
        "filter": "centered_so3_tangent_triangular",
        "radius": args.radius,
        "weights_at_full_window": list(
            range(1, args.radius + 2)
        ) + list(range(args.radius, 0, -1)),
        "target_reads": 0,
        "joint_support": "a1_changed_joints_intersect_cache_refine_mask",
        "frames": total_frames,
        "clips": summaries,
        "manifest_sha256": sha256(args.manifest),
        "source_predictions_manifest_sha256": sha256(
            source_manifest_path
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"clips": len(summaries), "frames": total_frames}, indent=2))


if __name__ == "__main__":
    main()
