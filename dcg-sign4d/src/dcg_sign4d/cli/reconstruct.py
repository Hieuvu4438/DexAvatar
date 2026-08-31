"""Audit and run the frozen DCG-Sign4D reconstruction contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from dcg_sign4d.data.manifest import load_manifest
from dcg_sign4d.inference.artifacts import (
    validate_prediction_artifact,
    write_prediction_artifact,
)
from dcg_sign4d.inference.provenance import build_run_identity
from dcg_sign4d.inference.readiness import audit_reconstruction_readiness
from dcg_sign4d.inference.runtime import ReconstructionConfig, ReconstructionRuntime
from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--readiness-report")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="commit each clip atomically and resume only hash-valid completed clips",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    output = Path(args.output)
    report_path = (
        Path(args.readiness_report)
        if args.readiness_report
        else output.parent / f"{output.name}_READINESS.json"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("reconstruction config must be a mapping")
    report = audit_reconstruction_readiness(config, manifest_path)
    report.update(
        {
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "requested_output": str(output.resolve()),
            "output_created": False,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", "utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))
    if report["status"] != "READY":
        raise RuntimeError(
            "reconstruction readiness is BLOCKED; no prediction output was created; "
            f"see {report_path}"
        )
    runtime_config = ReconstructionConfig.model_validate(config)
    all_items = load_manifest(manifest_path, require_existing_video=False)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index/count must satisfy 0 <= index < count")
    items = all_items[args.shard_index :: args.shard_count]
    report.update(
        {
            "manifest_total_clips": len(all_items),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "shard_clips": len(items),
        }
    )
    if output.exists() and not args.resume:
        raise FileExistsError(f"immutable prediction output exists: {output}")
    if args.resume:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    completed_clips: list[str] = []
    skipped_clips: list[str] = []
    try:
        runtime = ReconstructionRuntime(runtime_config, device=args.device)
        dependency_payload = yaml.safe_load(runtime_config.third_party_manifest.read_text("utf-8"))
        dependency_commits = {
            row["name"]: row["commit"] for row in dependency_payload["repositories"]
        }
        checkpoint_hashes = {
            "contact": file_sha256(runtime_config.contact.checkpoint / "weights.pt"),
            "diffusion": file_sha256(runtime_config.diffusion.checkpoint / "weights.pt"),
            "ranker": file_sha256(runtime_config.ranking.artifact),
        }
        for item in items:
            existing = temporary / item.clip_id
            if existing.exists():
                if not args.resume:
                    raise FileExistsError(f"immutable prediction exists: {existing}")
                validation = validate_prediction_artifact(existing)
                if validation["clip_id"] != item.clip_id:
                    raise ValueError(f"{item.clip_id}: existing artifact clip identity mismatch")
                completed_clips.append(item.clip_id)
                skipped_clips.append(item.clip_id)
                print(
                    json.dumps(
                        {
                            "clip_id": item.clip_id,
                            "completed": len(completed_clips),
                            "total": len(items),
                            "status": "resume_skip_valid",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            started = datetime.now(UTC)
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            inputs, hypotheses = runtime.reconstruct_clip(item.clip_id)
            ended = datetime.now(UTC)
            identity = build_run_identity(
                scope_root=Path(__file__).resolve().parents[3],
                config_path=config_path,
                manifest_path=manifest_path,
                dependency_commits=dependency_commits,
                checkpoint_sha256=checkpoint_hashes,
                sampler={
                    "diffusion_steps": runtime_config.inference.diffusion_steps,
                    "rounds": runtime_config.inference.rounds,
                    "num_hypotheses": runtime_config.inference.num_hypotheses,
                },
                started_at_utc=started.isoformat(),
                ended_at_utc=ended.isoformat(),
                peak_memory_bytes=(
                    int(torch.cuda.max_memory_allocated(runtime.device))
                    if runtime.device.type == "cuda"
                    else 0
                ),
                frame_count=item.effective_frame_count,
                execution_device=str(runtime.device),
                development_only=runtime_config.experiment.development_only,
            )
            manifest_payload = item.model_dump(mode="json")
            manifest_payload["manifest_sha256"] = file_sha256(manifest_path)
            ranker_config = {
                "fit_split": runtime.ranker_payload["fit_split"],
                "uses_ground_truth": runtime.ranker_payload["use_ground_truth"],
                "weights": runtime.ranker_payload["weights"],
                "artifact_sha256": file_sha256(runtime_config.ranking.artifact),
            }
            write_prediction_artifact(
                temporary,
                item.clip_id,
                (
                    inputs.trajectory,
                    inputs.camera,
                    inputs.initialization_metadata,
                ),
                hypotheses,
                identity,
                input_manifest=manifest_payload,
                observation_hashes=inputs.observation_hashes,
                ranker_config=ranker_config,
            )
            completed_clips.append(item.clip_id)
            report.update(
                {
                    "run_status": "IN_PROGRESS",
                    "completed_clips": len(completed_clips),
                    "total_clips": len(items),
                    "last_completed_clip": item.clip_id,
                    "resumable": args.resume,
                }
            )
            report_path.write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n", "utf-8"
            )
            print(
                json.dumps(
                    {
                        "clip_id": item.clip_id,
                        "completed": len(completed_clips),
                        "total": len(items),
                        "status": "completed",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        marker = (
            "RUN_COMPLETE"
            if args.shard_count == 1
            else f"SHARD_{args.shard_index:03d}_OF_{args.shard_count:03d}_COMPLETE"
        )
        (temporary / marker).write_text("complete\n", "utf-8")
        if not args.resume:
            os.replace(temporary, output)
    except Exception as exc:
        report.update(
            {
                "run_status": "INTERRUPTED",
                "completed_clips": len(completed_clips),
                "total_clips": len(items),
                "resumable": args.resume,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", "utf-8")
        if not args.resume:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    report["output_created"] = True
    report["clips_reconstructed"] = len(items)
    report["run_status"] = "COMPLETE"
    report["resumable"] = args.resume
    report["skipped_valid_clips"] = skipped_clips
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", "utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
