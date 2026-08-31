#!/usr/bin/env python3
"""Report read-only progress for the full PHOENIX/SOKE experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROGRESS = re.compile(
    r"\[wilor\]\s+(?P<done>\d+)/(?P<total>\d+) frames "
    r"\((?P<fps>[0-9.]+) fps\)"
)


def _session_alive(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _live_h32_workers() -> list[int]:
    workers = set()
    for command_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            tokens = command_path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        values = [token.decode(errors="replace") for token in tokens if token]
        if not any(token.endswith("extract_phoenix.py") for token in values):
            continue
        try:
            index = values.index("--worker_id")
            workers.add(int(values[index + 1]))
        except (ValueError, IndexError):
            continue
    return sorted(workers)


def _h32_worker_progress(
    *,
    video_root: Path,
    h32_root: Path,
    required_names: list[str],
    required_frame_counts: dict[str, int] | None = None,
    num_workers: int,
    recent_window_seconds: float,
    now: float,
) -> dict:
    """Report progress using the extractor's exact sorted-modulo sharding."""
    video_ids = sorted({path.stem for path in video_root.glob("*/*.mp4")})
    owner = {name: index % num_workers for index, name in enumerate(video_ids)}
    unknown = sorted({name for name in required_names if name not in owner})
    frame_counts = required_frame_counts or {}
    unknown_frame_counts = sorted(
        name
        for name in required_names
        if name not in frame_counts or int(frame_counts[name]) <= 0
    )
    workers = []
    for worker in range(num_workers):
        names = [name for name in required_names if owner.get(name) == worker]
        paths = [h32_root / f"{name}.pkl" for name in names]
        complete_paths = [path for path in paths if path.is_file()]
        recent = sum(
            now - path.stat().st_mtime <= recent_window_seconds
            for path in complete_paths
        )
        rate = recent / recent_window_seconds * 3600.0
        missing = len(paths) - len(complete_paths)
        declared_frames = sum(int(frame_counts.get(name, 0)) for name in names)
        complete_frames = sum(
            int(frame_counts.get(path.stem, 0)) for path in complete_paths
        )
        recent_frames = sum(
            int(frame_counts.get(path.stem, 0))
            for path in complete_paths
            if now - path.stat().st_mtime <= recent_window_seconds
        )
        missing_frames = declared_frames - complete_frames
        frame_rate = recent_frames / recent_window_seconds * 3600.0
        clip_eta = missing / rate if rate > 0.0 else None
        frame_eta = (
            missing_frames / frame_rate
            if not unknown_frame_counts and frame_rate > 0.0
            else None
        )
        workers.append(
            {
                "worker": worker,
                "declared_clips": len(paths),
                "complete_clips": len(complete_paths),
                "missing_clips": missing,
                "recent_complete_clips": recent,
                "observed_clips_per_hour": rate,
                "declared_source_frames": declared_frames,
                "complete_source_frames": complete_frames,
                "missing_source_frames": missing_frames,
                "recent_complete_source_frames": recent_frames,
                "observed_source_frames_per_hour": frame_rate,
                "clip_rate_estimated_remaining_hours": clip_eta,
                "estimated_remaining_hours": frame_eta or clip_eta,
                "eta_basis": (
                    "source_video_frames" if frame_eta is not None else "clips"
                ),
                "youngest_output_age_seconds": (
                    min(now - path.stat().st_mtime for path in complete_paths)
                    if complete_paths
                    else None
                ),
            }
        )
    finite_etas = [
        float(item["estimated_remaining_hours"])
        for item in workers
        if item["estimated_remaining_hours"] is not None
    ]
    return {
        "sharding": "sorted-video-id-modulo",
        "num_workers": num_workers,
        "unknown_required_clips": unknown,
        "unknown_required_frame_counts": unknown_frame_counts,
        "workers": workers,
        "critical_estimated_remaining_hours": (
            max(finite_etas) if len(finite_etas) == num_workers else None
        ),
    }


def _h32_segment_progress(
    *,
    video_root: Path,
    h32_root: Path,
    required_names: list[str],
    required_frame_counts: dict[str, int],
    segments: list[dict],
    recent_window_seconds: float,
    now: float,
) -> dict:
    """Report exact progress for independently bounded extractor slices."""
    video_ids = sorted({path.stem for path in video_root.glob("*/*.mp4")})
    required = set(required_names)
    covered: list[str] = []
    results = []
    for spec in segments:
        worker = int(spec["worker"])
        num_workers = int(spec["num_workers"])
        start = int(spec.get("assigned_start", 0))
        stop_value = spec.get("assigned_stop")
        stop = int(stop_value) if stop_value is not None else None
        partition = [
            name
            for index, name in enumerate(video_ids)
            if index % num_workers == worker
        ]
        assigned = partition[start:stop]
        names = [name for name in assigned if name in required]
        covered.extend(names)
        paths = [h32_root / f"{name}.pkl" for name in names]
        complete_paths = [path for path in paths if path.is_file()]
        recent_paths = [
            path
            for path in complete_paths
            if now - path.stat().st_mtime <= recent_window_seconds
        ]
        declared_frames = sum(required_frame_counts[name] for name in names)
        complete_frames = sum(
            required_frame_counts[path.stem] for path in complete_paths
        )
        recent_frames = sum(
            required_frame_counts[path.stem] for path in recent_paths
        )
        missing_frames = declared_frames - complete_frames
        clip_rate = len(recent_paths) / recent_window_seconds * 3600.0
        frame_rate = recent_frames / recent_window_seconds * 3600.0
        results.append(
            {
                "segment": str(spec["segment"]),
                "worker": worker,
                "num_workers": num_workers,
                "assigned_start": start,
                "assigned_stop": stop,
                "assigned_all_split_clips": len(assigned),
                "declared_required_clips": len(names),
                "complete_required_clips": len(complete_paths),
                "missing_required_clips": len(paths) - len(complete_paths),
                "recent_complete_required_clips": len(recent_paths),
                "observed_required_clips_per_hour": clip_rate,
                "declared_required_source_frames": declared_frames,
                "complete_required_source_frames": complete_frames,
                "missing_required_source_frames": missing_frames,
                "recent_complete_required_source_frames": recent_frames,
                "observed_required_source_frames_per_hour": frame_rate,
                "estimated_remaining_hours": (
                    missing_frames / frame_rate if frame_rate > 0.0 else None
                ),
            }
        )
    counts = Counter(covered)
    overlap = sorted(name for name, count in counts.items() if count > 1)
    missing = sorted(required - set(covered))
    finite = [
        float(item["estimated_remaining_hours"])
        for item in results
        if item["estimated_remaining_hours"] is not None
    ]
    return {
        "sharding": "sorted-video-id-modulo-with-bounded-slices",
        "segments": results,
        "declared_required_clips": len(required),
        "covered_required_clips": len(set(covered)),
        "missing_required_clips": missing,
        "overlap_required_clips": overlap,
        "critical_estimated_remaining_hours": (
            max(finite) if len(finite) == len(results) else None
        ),
    }


def _last_wilor_progress(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    matches = list(PROGRESS.finditer(log_path.read_text(errors="replace")))
    if not matches:
        return None
    values = matches[-1].groupdict()
    return {
        "frames": int(values["done"]),
        "total_frames": int(values["total"]),
        "fps": float(values["fps"]),
        "log": str(log_path),
    }


def _wilor_split(run_root: Path, split: str) -> dict:
    shard_root = run_root / "wilor_shards" / split
    report_path = shard_root / "shard_report.json"
    if not report_path.is_file():
        return {"declared_shards": 0, "complete_shards": 0, "running": []}
    report = json.loads(report_path.read_text())
    output_root = run_root / "wilor_outputs" / split
    completed_frames = 0
    completed_worker_fps = []
    complete = 0
    running = []
    for index, shard in enumerate(report["shards"]):
        destination = output_root / f"shard_{index:04d}"
        final_artifact = destination / "wilor" / "wilor.pkl"
        if final_artifact.is_file():
            complete += 1
            completed_frames += int(shard["frames"])
            completed_progress = _last_wilor_progress(
                run_root / "logs" / f"wilor_{split}_shard_{index:04d}.log"
            )
            if completed_progress and completed_progress["fps"] > 0.0:
                completed_worker_fps.append(float(completed_progress["fps"]))
        elif destination.exists():
            item = {"shard": index}
            progress = _last_wilor_progress(
                run_root / "logs" / f"wilor_{split}_shard_{index:04d}.log"
            )
            if progress:
                item.update(progress)
            running.append(item)
    running_frames = sum(int(item.get("frames", 0)) for item in running)
    aggregate_fps = sum(float(item.get("fps", 0.0)) for item in running)
    historical_worker_fps = (
        float(statistics.median(completed_worker_fps))
        if completed_worker_fps
        else 0.0
    )
    estimated_concurrency = len(running)
    stable_aggregate_fps = (
        historical_worker_fps * estimated_concurrency
        if historical_worker_fps > 0.0 and estimated_concurrency > 0
        else aggregate_fps
    )
    declared_frames = int(report["total_frames"])
    remaining_frames = max(
        0, declared_frames - completed_frames - running_frames
    )
    return {
        "declared_shards": len(report["shards"]),
        "complete_shards": complete,
        "declared_frames": declared_frames,
        "complete_frames": completed_frames,
        "running_frames": running_frames,
        "remaining_frames": remaining_frames,
        "observed_aggregate_fps": aggregate_fps,
        "historical_median_fps_per_worker": historical_worker_fps,
        "eta_aggregate_fps": stable_aggregate_fps,
        "estimated_remaining_hours": (
            remaining_frames / stable_aggregate_fps / 3600.0
            if stable_aggregate_fps > 0.0
            else None
        ),
        "running": running,
    }


def _gpu() -> dict | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode or not result.stdout.strip():
        return None
    fields = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    return {
        "index": int(fields[0]),
        "memory_used_mib": int(fields[1]),
        "memory_free_mib": int(fields[2]),
        "utilization_percent": int(fields[3]),
        "temperature_c": int(fields[4]),
    }


def _wilor_audit_status(path: Path, now: float) -> dict:
    if not path.is_file():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    verified = payload.get("verified", {})
    return {
        "exists": True,
        "schema": payload.get("schema"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "age_seconds": max(0.0, now - path.stat().st_mtime),
        "all_verified": bool(payload.get("all_verified", False)),
        "splits": payload.get("splits", {}),
        "verified_shards": len(verified),
        "verified_records": sum(
            int(item.get("records", 0)) for item in verified.values()
        ),
        "hamer_dropouts": sum(
            int(item.get("hamer_dropouts", 0)) for item in verified.values()
        ),
    }


def _h32_incremental_audit_status(path: Path, now: float) -> dict:
    if not path.is_file():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "schema": payload.get("schema"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "age_seconds": max(0.0, now - path.stat().st_mtime),
        "all_verified": bool(payload.get("all_verified", False)),
        "declared_clips": int(payload.get("declared_clips", 0)),
        "verified_clips": int(payload.get("verified_clips", 0)),
        "pending_clips": int(payload.get("pending_clips", 0)),
        "unstable_clips": len(payload.get("unstable_clips", [])),
        "newly_validated": int(payload.get("newly_validated", 0)),
        "reused": int(payload.get("reused", 0)),
        "h32_verified_content_set_sha256": payload.get(
            "h32_verified_content_set_sha256"
        ),
        "splits": payload.get("splits", {}),
    }


def report(project: Path) -> dict:
    project = project.resolve()
    run_root = project / "outputs" / "phoenix_soke_full_v1"
    selection_root = run_root / "selections"
    h32_root = (
        project
        / "data/SignAvatars/datasets/language2motion/annotations/SMPL-X_phoenix"
    )
    h32 = {}
    required_train_dev_names = []
    required_train_dev_frame_counts: dict[str, int] = {}
    required_train_dev_paths = []
    phoenix_video_root = None
    for split in ("train", "dev", "test"):
        selection = json.loads(
            (selection_root / split / "selection.json").read_text()
        )
        names = [str(item["source_clip"]) for item in selection["clips"]]
        if phoenix_video_root is None and selection["clips"]:
            phoenix_video_root = Path(selection["clips"][0]["video"]).parents[1]
        if split in {"train", "dev"}:
            required_train_dev_names.extend(names)
            required_train_dev_paths.extend(h32_root / f"{name}.pkl" for name in names)
            for item in selection["clips"]:
                name = str(item["source_clip"])
                frames = int(item["source_contract"]["frame_count"])
                if name in required_train_dev_frame_counts:
                    raise ValueError(f"Duplicate required H32 source clip: {name}")
                required_train_dev_frame_counts[name] = frames
        complete = sum((h32_root / f"{name}.pkl").is_file() for name in names)
        h32[split] = {
            "declared_clips": len(names),
            "complete_clips": complete,
            "missing_clips": len(names) - complete,
        }
    now = time.time()
    h32_window_seconds = 600.0
    recent_h32 = sum(
        path.is_file() and now - path.stat().st_mtime <= h32_window_seconds
        for path in required_train_dev_paths
    )
    required_complete = sum(path.is_file() for path in required_train_dev_paths)
    required_missing = len(required_train_dev_paths) - required_complete
    h32_rate_per_hour = recent_h32 / h32_window_seconds * 3600.0
    required_source_frames = sum(required_train_dev_frame_counts.values())
    complete_source_frames = sum(
        required_train_dev_frame_counts[path.stem]
        for path in required_train_dev_paths
        if path.is_file()
    )
    recent_source_frames = sum(
        required_train_dev_frame_counts[path.stem]
        for path in required_train_dev_paths
        if path.is_file() and now - path.stat().st_mtime <= h32_window_seconds
    )
    missing_source_frames = required_source_frames - complete_source_frames
    h32_frame_rate_per_hour = (
        recent_source_frames / h32_window_seconds * 3600.0
    )
    if phoenix_video_root is None:
        raise ValueError("No PHOENIX videos declared by the selections")
    h32_by_worker = _h32_worker_progress(
        video_root=phoenix_video_root,
        h32_root=h32_root,
        required_names=required_train_dev_names,
        required_frame_counts=required_train_dev_frame_counts,
        num_workers=2,
        recent_window_seconds=h32_window_seconds,
        now=now,
    )
    h32_active_segments = _h32_segment_progress(
        video_root=phoenix_video_root,
        h32_root=h32_root,
        required_names=required_train_dev_names,
        required_frame_counts=required_train_dev_frame_counts,
        segments=[
            {
                "segment": "worker0_full",
                "worker": 0,
                "num_workers": 2,
            },
            {
                "segment": "worker1_lower",
                "worker": 1,
                "num_workers": 2,
                "assigned_stop": 2725,
            },
            {
                "segment": "worker1_tail",
                "worker": 1,
                "num_workers": 2,
                "assigned_start": 2725,
            },
        ],
        recent_window_seconds=h32_window_seconds,
        now=now,
    )
    disk = shutil.disk_usage(project)
    train_root = (
        project / "outputs/phase2r/phoenix_soke_full_raw_fusion_v1_seed42"
    )
    cache_root = project / "cache/signal4d_external"
    cache_status = {
        "train": (
            cache_root / "phoenix_soke_full_train_v1/splits/train.json"
        ).is_file(),
        "dev": (
            cache_root / "phoenix_soke_full_dev_v1/splits/val.json"
        ).is_file(),
        "test": (
            cache_root / "phoenix_soke_full_test_v1/splits/test.json"
        ).is_file(),
    }
    training_status = {
        "best_checkpoint": (train_root / "best.pt").is_file(),
        "last_checkpoint": (train_root / "last.pt").is_file(),
        "dev_calibration": (
            train_root / "phoenix_dev_benefit_calibration.json"
        ).is_file(),
        "test_evaluation": (
            train_root / "phoenix_test_soke_pampjpe.json"
        ).is_file(),
        "test_report": (
            train_root / "phoenix_test_soke_pampjpe.md"
        ).is_file(),
    }
    if training_status["test_evaluation"]:
        stage = "complete" if training_status["test_report"] else "render_report"
    elif training_status["dev_calibration"]:
        stage = "test_frontend_cache_and_evaluation"
    elif training_status["best_checkpoint"] or training_status["last_checkpoint"]:
        stage = "transformer_training_or_checkpoint_selection"
    elif cache_status["train"] and cache_status["dev"]:
        stage = "transformer_training_pending_or_starting"
    elif cache_status["train"] or cache_status["dev"]:
        stage = "train_dev_cache_materialization"
    else:
        stage = "train_dev_frontend_extraction"
    return {
        "schema": "signal4d-phoenix-soke-progress-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "sessions": {
            "pipeline": _session_alive("phoenix_soke_transformer_full"),
            "h32_worker1_resume": _session_alive("phoenix_h32_worker1_resume"),
            "h32_worker1_tail": _session_alive("phoenix_h32_worker1_tail"),
            "h32_audit_watch": _session_alive("phoenix_h32_audit_watch"),
            "h32_incremental_audit_watch": _session_alive(
                "phoenix_h32_incremental_audit_watch"
            ),
            "wilor_audit_watch": _session_alive("phoenix_wilor_audit_watch"),
            "progress_watch": _session_alive("phoenix_progress_watch"),
        },
        "live_h32_workers": _live_h32_workers(),
        "h32": h32,
        "h32_by_worker": h32_by_worker,
        "h32_active_segments": h32_active_segments,
        "h32_incremental_audit": _h32_incremental_audit_status(
            run_root / "h32_incremental_audit.json", now
        ),
        "h32_required_train_dev": {
            "declared_clips": len(required_train_dev_paths),
            "complete_clips": required_complete,
            "missing_clips": required_missing,
            "recent_window_seconds": h32_window_seconds,
            "recent_complete_clips": recent_h32,
            "observed_clips_per_hour": h32_rate_per_hour,
            "declared_source_frames": required_source_frames,
            "complete_source_frames": complete_source_frames,
            "missing_source_frames": missing_source_frames,
            "recent_complete_source_frames": recent_source_frames,
            "observed_source_frames_per_hour": h32_frame_rate_per_hour,
            "aggregate_rate_estimated_remaining_hours": (
                required_missing / h32_rate_per_hour
                if h32_rate_per_hour > 0.0
                else None
            ),
            "aggregate_frame_rate_estimated_remaining_hours": (
                missing_source_frames / h32_frame_rate_per_hour
                if h32_frame_rate_per_hour > 0.0
                else None
            ),
            "estimated_remaining_hours": (
                h32_by_worker["critical_estimated_remaining_hours"]
            ),
        },
        "wilor": {
            split: _wilor_split(run_root, split)
            for split in ("train", "dev", "test")
        },
        "wilor_audit": _wilor_audit_status(
            run_root / "wilor_frontend_audit.json", now
        ),
        "cache": cache_status,
        "training": training_status,
        "gpu": _gpu(),
        "disk": {
            "free_gib": disk.free / 2**30,
            "total_gib": disk.total / 2**30,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continue reporting until the experiment reaches the complete stage.",
    )
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Append one compact, timestamped progress object per observation.",
    )
    arguments = parser.parse_args()
    if arguments.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    output = arguments.output_jsonl.resolve() if arguments.output_jsonl else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        payload = report(arguments.project)
        if output is None:
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        else:
            rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            with output.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
                handle.flush()
            print(
                json.dumps(
                    {
                        "timestamp_utc": payload["timestamp_utc"],
                        "stage": payload["stage"],
                        "h32_complete": payload["h32_required_train_dev"][
                            "complete_clips"
                        ],
                        "wilor_train_verified": payload["wilor_audit"].get(
                            "splits", {}
                        ).get("train", {}).get("verified_shards", 0),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not arguments.watch or payload["stage"] == "complete":
            break
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    main()
