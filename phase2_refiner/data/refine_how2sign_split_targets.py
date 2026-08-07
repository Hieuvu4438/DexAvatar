"""Create independent 2D-temporal targets for one append-only How2Sign split."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip
from phase2_refiner.data.refine_how2sign_targets import (
    INITIALIZER_PROVIDER,
    REGIONS,
    TARGET_PROVIDER,
    _fit_batch,
    _teacher_observations,
    _write_manifest,
)
from phase2_refiner.render import create_smplx_model


def build(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only output already exists: {output}")
    paths = _manifest_paths(args.input_manifest.resolve())
    if args.max_clips > 0:
        paths = paths[: args.max_clips]
    if not paths:
        raise ValueError("Input manifest has no clips")
    output.mkdir(parents=True)
    (output / "splits").mkdir()
    clip_dir = output / "clips" / args.split
    clip_dir.mkdir(parents=True)
    model = create_smplx_model(args.model_folder.resolve(), torch.device(args.device))
    model.requires_grad_(False)
    entries: list[str] = []
    groups: set[str] = set()
    reports: list[dict] = []
    try:
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            clips = [load_cache_clip(path) for path in batch_paths]
            teacher = [_teacher_observations(clip) for clip in clips]
            targets, batch_reports = _fit_batch(
                model,
                clips,
                teacher,
                torch.device(args.device),
                args.iterations,
                args.learning_rate,
                args.body_max_degrees,
                args.hand_max_degrees,
            )
            for clip, observations, target, report in zip(
                clips, teacher, targets, batch_reports, strict=True
            ):
                report["clip_id"] = clip.clip_id
                reports.append(report)
                if not report["accepted"]:
                    continue
                metadata = json.loads(clip.metadata_json)
                group = str(metadata["source_group"])
                metadata.update(
                    {
                        "target_type": "independent_pseudo_target",
                        "initializer_expert": INITIALIZER_PROVIDER,
                        "target_provider": TARGET_PROVIDER,
                        "independent_refinement_signal": (
                            "ordered How2Sign 133-point 2D tracks plus temporal constraints"
                        ),
                        "initializer_matches_locked_lane_a1": False,
                        "target_quality": report,
                    }
                )
                refined = replace(
                    clip,
                    target_axis_angle=target.astype(np.float32),
                    target_rotation_valid=np.ones_like(
                        clip.target_rotation_valid, dtype=bool
                    ),
                    keypoints_2d=observations[0],
                    keypoint_valid=observations[2],
                    observation_features=observations[3],
                    u0_reliability=(observations[1] * observations[2]).astype(
                        np.float32
                    ),
                    metadata_json=json.dumps(metadata, sort_keys=True),
                )
                destination = clip_dir / f"{clip.clip_id}.npz"
                save_cache_clip(destination, refined)
                entries.append(f"../clips/{args.split}/{destination.name}")
                groups.add(group)
            print(
                f"[2d-temporal-single] split={args.split} "
                f"processed={min(start + len(clips), len(paths))}/{len(paths)} "
                f"accepted={len(entries)}",
                flush=True,
            )
        if not entries:
            raise RuntimeError("No clips passed target quality filters")
        _write_manifest(output, args.split, entries, groups)
        accepted = [item for item in reports if item["accepted"]]
        summary = {
            "schema_version": 1,
            "official_split": args.split,
            "method": TARGET_PROVIDER,
            "initializer": INITIALIZER_PROVIDER,
            "requested": len(paths),
            "accepted": len(accepted),
            "rejected": len(reports) - len(accepted),
            "frames": len(accepted) * len(clips[0].frame_names),
            "source_groups": len(groups),
            "parameters": {
                "iterations": args.iterations,
                "learning_rate": args.learning_rate,
                "body_max_degrees": args.body_max_degrees,
                "hand_max_degrees": args.hand_max_degrees,
                "batch_size": args.batch_size,
            },
            "mean_relative_gain": {
                region: float(np.mean([r["relative_gain"][region] for r in accepted]))
                for region in ("all", *REGIONS)
            },
        }
        (output / "refinement_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (output / f"{args.split}_per_clip.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:
            for report in reports:
                handle.write(json.dumps(report, sort_keys=True) + "\n")
        return summary
    except Exception:
        shutil.rmtree(output)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--model-folder", type=Path, default=Path("SMPLer-X/common/utils/human_model_files"))
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--body-max-degrees", type=float, default=12.0)
    parser.add_argument("--hand-max-degrees", type=float, default=18.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
