"""Run the frozen Lane-L A1 stack on the exact frames in a Phase-2 cache.

The locked expert pipeline consumes image directories, while How2Sign cache
entries bind each sample to ``video.mp4#frame=N``.  This wrapper bridges those
contracts without modifying the frozen pipeline: it extracts only the bound
frames, runs the exact stack in bounded batches, validates the complete result
schema, and atomically publishes per-clip result PKLs.

Long runs are resumable and append-only.  A complete, valid clip is skipped;
an existing incomplete or invalid clip fails closed.  Failed batch workspaces
are retained for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import CacheClip, load_cache_clip
from phase2_refiner.data.materialize_exact_a1_cache import (
    STACK_ID,
    validate_exact_a1_provenance,
    verify_exact_a1_components,
)
from phase2_refiner.provenance import sha256_file


REQUIRED_RESULT_SHAPES = {
    "global_orient": 3,
    "body_pose": 63,
    "left_hand_pose": 45,
    "right_hand_pose": 45,
    "jaw_pose": 3,
    "leye_pose": 3,
    "reye_pose": 3,
    "expression": 10,
    "betas": 10,
    "transl": 3,
}


@dataclass(frozen=True)
class ClipJob:
    cache_path: Path
    clip: CacheClip
    video_path: Path
    frame_indices: tuple[int, ...]

    @property
    def frames(self) -> int:
        return len(self.frame_indices)


def parse_video_frame_reference(reference: str) -> tuple[Path, int]:
    marker = "#frame="
    if reference.count(marker) != 1:
        raise ValueError(f"Invalid video-frame reference: {reference!r}")
    raw_path, raw_frame = reference.rsplit(marker, 1)
    try:
        frame = int(raw_frame)
    except ValueError as error:
        raise ValueError(f"Invalid frame number in {reference!r}") from error
    if frame < 0:
        raise ValueError(f"Negative frame number in {reference!r}")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"How2Sign source video is missing: {path}")
    return path, frame


def validate_exact_result_payload(payload: Any, source: Path | str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Exact-A1 result is not a dictionary: {source}")
    for key, size in REQUIRED_RESULT_SHAPES.items():
        if key not in payload:
            raise ValueError(f"Exact-A1 result {source} is missing {key}")
        array = np.asarray(payload[key])
        if array.size != size or not np.isfinite(array).all():
            raise ValueError(
                f"Exact-A1 result {source} has invalid {key}: "
                f"shape={array.shape}, expected_elements={size}"
            )
    if "K" not in payload:
        raise ValueError(f"Exact-A1 result {source} is missing K")
    camera = np.asarray(payload["K"])
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError(
            f"Exact-A1 result {source} has invalid K shape={camera.shape}"
        )


def validate_exact_result_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Exact-A1 result file is missing: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    validate_exact_result_payload(payload, path)
    return sha256_file(path)


def _clip_job(path: Path) -> ClipJob:
    clip = load_cache_clip(path)
    if len(clip.source_paths) != len(clip.frame_names):
        raise ValueError(f"Source/frame count mismatch in {path}")
    parsed = [parse_video_frame_reference(str(item)) for item in clip.source_paths]
    videos = {item[0] for item in parsed}
    if len(videos) != 1:
        raise ValueError(f"One cache clip references multiple videos: {path}")
    indices = tuple(item[1] for item in parsed)
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate source frame in {path}")
    if tuple(sorted(indices)) != indices:
        raise ValueError(f"Source frames are not ordered in {path}")
    if len(clip.frame_names) != len(set(map(str, clip.frame_names))):
        raise ValueError(f"Duplicate output frame name in {path}")
    return ClipJob(path, clip, videos.pop(), indices)


def load_jobs(template_root: Path, exact_root: Path) -> tuple[list[ClipJob], int]:
    jobs: list[ClipJob] = []
    seen_clips: set[str] = set()
    seen_frames: set[str] = set()
    completed_frames = 0
    for split in ("train", "val", "calibration"):
        manifest = template_root / "splits" / f"{split}.json"
        for path in _manifest_paths(manifest):
            job = _clip_job(path)
            clip_id = job.clip.clip_id
            if clip_id in seen_clips:
                raise ValueError(f"Duplicate clip ID across manifests: {clip_id}")
            seen_clips.add(clip_id)
            overlap = seen_frames.intersection(map(str, job.clip.frame_names))
            if overlap:
                raise ValueError(f"Duplicate output frame name: {sorted(overlap)[0]}")
            seen_frames.update(map(str, job.clip.frame_names))
            destination = exact_root / clip_id / "smplifyx" / "results"
            if destination.exists():
                hashes = validate_clip_results(job, destination)
                completed_frames += len(hashes)
                continue
            jobs.append(job)
    return jobs, completed_frames


def validate_clip_results(job: ClipJob, results: Path) -> dict[str, str]:
    expected = {f"{name}.pkl" for name in map(str, job.clip.frame_names)}
    actual = {path.name for path in results.glob("*.pkl") if path.is_file()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Exact-A1 clip coverage failure for {job.clip.clip_id}: "
            f"expected={len(expected)} actual={len(actual)} "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    return {
        name: validate_exact_result_file(results / name) for name in sorted(expected)
    }


def _batches(jobs: list[ClipJob], frame_limit: int) -> list[list[ClipJob]]:
    if frame_limit < 1:
        raise ValueError("--batch-frames must be positive")
    batches: list[list[ClipJob]] = []
    current: list[ClipJob] = []
    frames = 0
    for job in jobs:
        if current and frames + job.frames > frame_limit:
            batches.append(current)
            current, frames = [], 0
        current.append(job)
        frames += job.frames
    if current:
        batches.append(current)
    return batches


def extract_job_frames(job: ClipJob, image_root: Path) -> None:
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(job.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {job.video_path}")
    targets = dict(zip(job.frame_indices, map(str, job.clip.frame_names), strict=True))
    written = 0
    try:
        for index in range(job.frame_indices[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Video ended before frame {index}: {job.video_path}"
                )
            name = targets.get(index)
            if name is None:
                continue
            destination = image_root / f"{name}.png"
            if not cv2.imwrite(
                str(destination), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
            ):
                raise RuntimeError(f"Failed to write extracted frame: {destination}")
            written += 1
    finally:
        capture.release()
    if written != job.frames:
        raise RuntimeError(
            f"Extracted {written}/{job.frames} frames for {job.clip.clip_id}"
        )


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"unix_time": time.time(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def limit_cpu_affinity(cpu_threads: int) -> tuple[int, ...]:
    """Restrict this process and every future expert child to at most N CPUs."""
    if not hasattr(os, "sched_getaffinity"):
        return ()
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    selected = allowed[:cpu_threads]
    if not selected:
        raise RuntimeError("No CPUs are available to the exact-A1 process")
    os.sched_setaffinity(0, selected)
    return selected


def publish_batch(
    jobs: list[ClipJob], pipeline_results: Path, exact_root: Path, batch_id: str
) -> dict[str, dict[str, str]]:
    staging_root = exact_root / ".staging" / batch_id
    if staging_root.exists():
        raise FileExistsError(f"Append-only staging directory exists: {staging_root}")
    staging_root.mkdir(parents=True)
    published: dict[str, dict[str, str]] = {}
    try:
        for job in jobs:
            clip_stage = staging_root / job.clip.clip_id / "smplifyx" / "results"
            clip_stage.mkdir(parents=True)
            for frame_name in map(str, job.clip.frame_names):
                source = pipeline_results / f"{frame_name}.pkl"
                validate_exact_result_file(source)
                shutil.copy2(source, clip_stage / source.name)
            hashes = validate_clip_results(job, clip_stage)
            destination = exact_root / job.clip.clip_id
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite exact-A1 clip: {destination}"
                )
            os.replace(clip_stage.parents[1], destination)
            published[job.clip.clip_id] = hashes
    finally:
        if staging_root.exists() and not any(staging_root.rglob("*.pkl")):
            shutil.rmtree(staging_root)
    return published


def run(args: argparse.Namespace) -> dict[str, Any]:
    cpu_affinity = limit_cpu_affinity(args.cpu_threads)
    repository = Path(__file__).resolve().parents[2]
    template_root = args.template_root.resolve()
    exact_root = args.exact_root.resolve()
    work_root = args.work_root.resolve()
    pipeline = args.pipeline.resolve()
    fitting_experiment = args.fitting_experiment.resolve()
    with args.provenance.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    validate_exact_a1_provenance(provenance)
    verify_exact_a1_components(provenance)
    if not pipeline.is_file() or pipeline != (
        repository / "methods" / "Full_running_command_wilor_ensemble.sh"
    ).resolve():
        raise ValueError(f"Pipeline is not the frozen exact-A1 script: {pipeline}")
    if not fitting_experiment.is_dir():
        raise FileNotFoundError(f"Fitting experiment is missing: {fitting_experiment}")
    exact_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    jobs, completed_frames = load_jobs(template_root, exact_root)
    batches = _batches(jobs, args.batch_frames)
    total_pending_frames = sum(job.frames for job in jobs)
    report: dict[str, Any] = {
        "stack_id": STACK_ID,
        "component_provenance_sha256": sha256_file(args.provenance),
        "pending_clips": len(jobs),
        "pending_frames": total_pending_frames,
        "completed_frames_at_start": completed_frames,
        "batches": len(batches),
        "batch_frames": args.batch_frames,
        "cpu_affinity": cpu_affinity,
        "dry_run": args.dry_run,
    }
    print(json.dumps({"stage1_plan": report}, indent=2, sort_keys=True), flush=True)
    append_event(args.progress_log, {"event": "plan", **report})
    if args.dry_run:
        return report

    selected_batches = batches[: args.max_batches] if args.max_batches else batches
    processed_clips = processed_frames = 0
    for batch_index, jobs_in_batch in enumerate(selected_batches, start=1):
        # Recheck the immutable stack immediately before every expert execution.
        verify_exact_a1_components(provenance)
        batch_id = f"batch_{batch_index:06d}"
        batch_root = work_root / batch_id
        if batch_root.exists():
            raise FileExistsError(
                f"Retained batch workspace requires diagnosis: {batch_root}"
            )
        image_root = batch_root / "images"
        output_root = batch_root / "expert_output"
        image_root.mkdir(parents=True)
        output_root.mkdir()
        frames = sum(job.frames for job in jobs_in_batch)
        append_event(
            args.progress_log,
            {
                "event": "batch_start",
                "batch_id": batch_id,
                "clips": len(jobs_in_batch),
                "frames": frames,
            },
        )
        print(
            f"[exact-a1] {batch_id} extracting clips={len(jobs_in_batch)} "
            f"frames={frames}",
            flush=True,
        )
        for job in jobs_in_batch:
            extract_job_frames(job, image_root)
        environment = os.environ.copy()
        environment.update(
            {
                "ROOT_PATH": str(image_root),
                "OUTPUT_PATH": str(output_root),
                "FITTING_EXPERIMENT": str(fitting_experiment),
                "OMP_NUM_THREADS": str(args.cpu_threads),
                "MKL_NUM_THREADS": str(args.cpu_threads),
                "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
                "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
            }
        )
        print(f"[exact-a1] {batch_id} running frozen pipeline", flush=True)
        try:
            subprocess.run(
                ["bash", str(pipeline)],
                cwd=repository,
                env=environment,
                check=True,
            )
            results = output_root / "smplifyx" / "results"
            published = publish_batch(jobs_in_batch, results, exact_root, batch_id)
        except Exception as error:
            append_event(
                args.progress_log,
                {
                    "event": "batch_failure",
                    "batch_id": batch_id,
                    "error": repr(error),
                    "workspace": str(batch_root),
                },
            )
            raise
        processed_clips += len(published)
        processed_frames += sum(len(item) for item in published.values())
        shutil.rmtree(batch_root)
        append_event(
            args.progress_log,
            {
                "event": "batch_complete",
                "batch_id": batch_id,
                "processed_clips": processed_clips,
                "processed_frames": processed_frames,
            },
        )
        print(
            f"[exact-a1] {batch_id} complete total_clips={processed_clips} "
            f"total_frames={processed_frames}",
            flush=True,
        )

    report.update(
        {
            "processed_clips": processed_clips,
            "processed_frames": processed_frames,
            "run_complete": len(selected_batches) == len(batches),
        }
    )
    append_event(args.progress_log, {"event": "run_complete", **report})
    if report["run_complete"]:
        final_report = exact_root / "stage1_extraction_report.json"
        if final_report.exists():
            raise FileExistsError(f"Append-only Stage-1 report exists: {final_report}")
        with final_report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({"stage1_result": report}, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("methods/Full_running_command_wilor_ensemble.sh"),
    )
    parser.add_argument(
        "--fitting-experiment", type=Path, default=Path("dexavatar_fitting")
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=Path("logs/phase2/how2sign_exact_a1_stage1_progress.jsonl"),
    )
    parser.add_argument("--batch-frames", type=int, default=512)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
