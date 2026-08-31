"""Run frozen SMPLer-X H32 on a frame-exact sign-domain selection.

This is one component of the label-free Module-1 expert fusion.  The outputs
remain target-free and are later fused with independently extracted WiLoR hand
predictions before SOKE/SignAvatars targets are attached.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from phase2_refiner.data.extract_how2sign_teacher import (
    MODEL_NAME,
    _decode_frames,
    _infer,
    _load_model,
    _load_track,
)
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-sign-domain-smplerx-selection-v1"


def _load_selection(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "signal4d-sign-domain-selection-v1":
        raise ValueError(f"Unsupported selection schema: {path}")
    if not isinstance(payload.get("clips"), list) or not payload["clips"]:
        raise ValueError(f"Selection has no clips: {path}")
    return payload


def _take_by_dataset(
    entries: list[dict], max_soke: int, max_signavatars: int
) -> list[dict]:
    limits = {"SOKE": max_soke, "SignAvatars": max_signavatars}
    selected = []
    for dataset, maximum in limits.items():
        candidates = [entry for entry in entries if entry["dataset"] == dataset]
        selected.extend(candidates[:maximum] if maximum > 0 else candidates)
    return sorted(selected, key=lambda entry: (entry["dataset"], entry["clip_id"]))


def _existing_valid(path: Path, frame_indices: list[int]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                payload["body_pose"].shape == (len(frame_indices), 63)
                and np.array_equal(
                    payload["sample_indices"], np.asarray(frame_indices, dtype=np.int64)
                )
            )
    except Exception:
        return False


def run(args: argparse.Namespace) -> dict:
    source = args.selection.resolve()
    output = args.output.resolve()
    manifest = _load_selection(source)
    selected = _take_by_dataset(
        manifest["clips"], args.max_soke, args.max_signavatars
    )
    if not selected:
        raise ValueError("Component selection is empty")
    output.mkdir(parents=True, exist_ok=True)
    locked_selection = {
        "schema": SCHEMA,
        "source_selection": str(source),
        "source_selection_sha256": sha256_file(source),
        "split": manifest["split"],
        "teacher": MODEL_NAME,
        "clips": selected,
        "target_fields_read": False,
    }
    selection_path = output / "selection.json"
    rendered = json.dumps(locked_selection, indent=2, sort_keys=True) + "\n"
    if selection_path.exists():
        if selection_path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Existing locked selection differs: {selection_path}")
    else:
        selection_path.write_text(rendered, encoding="utf-8")
    clip_dir = output / "clips"
    clip_dir.mkdir(exist_ok=True)
    pending = [
        entry
        for entry in selected
        if not _existing_valid(
            clip_dir / f"{entry['clip_id']}.npz", entry["frame_indices"]
        )
    ]
    print(
        f"[smplerx-selection] split={manifest['split']} selected={len(selected)} "
        f"complete={len(selected)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return {
            "split": manifest["split"],
            "selected": len(selected),
            "completed": len(selected),
            "pending": 0,
        }

    model, cfg = _load_model(output)
    from utils.preprocessing import generate_patch_image, process_bbox

    def prepare(entry: dict) -> dict:
        keypoints, scores = _load_track(Path(entry["keypoint_path"]))
        indices = np.asarray(entry["frame_indices"], dtype=np.int64)
        patches, bboxes, image_size, fps = _decode_frames(
            Path(entry["video"]),
            indices,
            keypoints,
            process_bbox,
            generate_patch_image,
            cfg.input_img_shape,
        )
        return {
            "entry": entry,
            "patches": patches,
            "indices": indices,
            "keypoints": keypoints[indices],
            "scores": scores[indices],
            "bboxes": bboxes,
            "image_size": image_size,
            "fps": fps,
        }

    groups = [
        pending[start : start + args.clips_per_batch]
        for start in range(0, len(pending), args.clips_per_batch)
    ]
    started = time.time()
    completed_at_start = len(selected) - len(pending)
    completed = completed_at_start
    failures = output / "failures.jsonl"
    with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        for group_index, group in enumerate(groups, start=1):
            futures = [(entry, executor.submit(prepare, entry)) for entry in group]
            decoded = []
            for entry, future in futures:
                try:
                    decoded.append(future.result())
                except Exception as error:
                    with failures.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"clip_id": entry["clip_id"], "error": repr(error)},
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    raise
            patches = np.concatenate([item["patches"] for item in decoded], axis=0)
            prediction = _infer(model, patches, args.batch_size, False)
            offset = 0
            for item in decoded:
                frames = len(item["indices"])
                end = offset + frames
                payload = {
                    **{key: value[offset:end] for key, value in prediction.items()},
                    "sample_indices": item["indices"],
                    "keypoints_2d": item["keypoints"],
                    "keypoint_scores": item["scores"],
                    "bboxes": item["bboxes"],
                    "image_size": item["image_size"],
                    "fps": np.asarray(item["fps"], dtype=np.float32),
                }
                destination = clip_dir / f"{item['entry']['clip_id']}.npz"
                temporary = destination.with_suffix(".npz.partial")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **payload)
                os.replace(temporary, destination)
                offset = end
                completed += 1
            elapsed = time.time() - started
            rate = (completed - completed_at_start) / max(elapsed, 1e-6)
            eta = (len(selected) - completed) / max(rate, 1e-6)
            print(
                f"[smplerx-progress] group={group_index}/{len(groups)} "
                f"clips={completed}/{len(selected)} rate={rate:.3f}_clips/s "
                f"eta_hours={eta/3600:.2f} "
                f"gpu_peak_gib={torch.cuda.max_memory_allocated()/2**30:.2f}",
                flush=True,
            )
    if completed != len(selected):
        raise RuntimeError(f"Incomplete SMPLer-X extraction: {completed}/{len(selected)}")
    report = {
        "schema": SCHEMA,
        "split": manifest["split"],
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "clips": len(selected),
        "frames": sum(len(entry["frame_indices"]) for entry in selected),
        "teacher": MODEL_NAME,
        "target_fields_read": False,
    }
    report_path = output / "extraction_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-soke", type=int, default=0)
    parser.add_argument("--max-signavatars", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clips-per-batch", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    args = parser.parse_args()
    if min(
        args.max_soke,
        args.max_signavatars,
        args.batch_size,
        args.clips_per_batch,
        args.decode_workers,
    ) < 0 or min(args.batch_size, args.clips_per_batch, args.decode_workers) < 1:
        parser.error("Limits must be non-negative and batch/decode sizes positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))

