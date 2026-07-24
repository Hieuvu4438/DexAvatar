"""Extract ordered full SMPL-X pseudo-targets from How2Sign without copying frames.

This command must run in the ``smpler_x`` conda environment.  It uses the
official How2Sign train/dev directories only, selects clips deterministically,
decodes uniformly sampled source-video frames in memory, and uses the supplied
133-keypoint tracks to crop the signer.  Outputs are compact, resumable NPZ
files in a new directory; source videos and pose tracks are never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SMPLERX_MAIN = REPO_ROOT / "SMPLer-X" / "main"
SMPLERX_COMMON = REPO_ROOT / "SMPLer-X" / "common"
SMPLERX_DATA = REPO_ROOT / "SMPLer-X" / "data"
MODEL_NAME = "smpler_x_h32"
TARGET_KEYS = {
    "global_orient": "smplx_root_pose",
    "body_pose": "smplx_body_pose",
    "left_hand_pose": "smplx_lhand_pose",
    "right_hand_pose": "smplx_rhand_pose",
    "jaw_pose": "smplx_jaw_pose",
    "betas": "smplx_shape",
    "expression": "smplx_expr",
    "transl": "cam_trans",
}


def _split_paths(root: Path, split: str) -> tuple[Path, Path]:
    if split == "train":
        return root / "train" / "raw_videos", root / "train" / "train_pose"
    if split == "val":
        return root / "eval" / "raw_videos", root / "eval" / "eval_pose"
    raise ValueError("Only official train and val/dev splits are permitted")


def _hash_order(name: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def _load_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    keypoints = np.asarray(payload["keypoints"], dtype=np.float32)
    scores = np.asarray(payload["scores"], dtype=np.float32)
    if keypoints.ndim == 4 and keypoints.shape[1] == 1:
        keypoints = keypoints[:, 0]
    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        raise ValueError(f"Unexpected keypoints shape in {path}: {keypoints.shape}")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError(f"Unexpected scores shape in {path}: {scores.shape}")
    return keypoints, scores


def _selection(
    video_dir: Path,
    pose_dir: Path,
    max_clips: int,
    frames_per_clip: int,
    seed: int,
) -> list[dict]:
    candidates = []
    for pose_path in pose_dir.glob("*.pkl"):
        video_path = video_dir / f"{pose_path.stem}.mp4"
        if not video_path.is_file():
            continue
        try:
            keypoints, _ = _load_track(pose_path)
        except Exception:
            continue
        if len(keypoints) < frames_per_clip:
            continue
        candidates.append(
            {
                "clip_id": pose_path.stem,
                "frames": len(keypoints),
                "video": str(video_path.resolve()),
                "pose": str(pose_path.resolve()),
            }
        )
    candidates.sort(key=lambda item: _hash_order(item["clip_id"], seed))
    return candidates[:max_clips] if max_clips > 0 else candidates


def _write_selection(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Existing selection differs; use a new output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _load_model(output: Path):
    os.chdir(SMPLERX_MAIN)
    for path in (SMPLERX_MAIN, SMPLERX_COMMON, SMPLERX_DATA):
        sys.path.insert(0, str(path))
    from config import cfg

    config_path = SMPLERX_MAIN / "config" / f"config_{MODEL_NAME}.py"
    checkpoint = REPO_ROOT / "checkpoints" / f"{MODEL_NAME}.pth.tar"
    cfg.get_config_fromfile(str(config_path))
    cfg.update_test_config(
        "EHF",
        "agora_model",
        shapy_eval_split=None,
        pretrained_model_path=str(checkpoint),
        use_cache=False,
    )
    cfg.update_config(1, str(output / "smplerx_runtime"))
    torch.backends.cudnn.benchmark = True
    from base import Demoer

    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    return demoer.model, cfg


def _bbox_from_track(points: np.ndarray, width: int, height: int) -> np.ndarray:
    # COCO-WholeBody: 23 body/feet, then face, then both 21-point hands.
    selected = np.concatenate((points[:23], points[91:133]), axis=0).copy()
    selected[:, 0] *= width
    selected[:, 1] *= height
    inside = (
        np.isfinite(selected).all(axis=1)
        & (selected[:, 0] >= 0)
        & (selected[:, 0] < width)
        & (selected[:, 1] >= 0)
        & (selected[:, 1] < height)
    )
    selected = selected[inside]
    if len(selected) < 8:
        return np.asarray([0.15 * width, 0.02 * height, 0.70 * width, 0.96 * height])
    low = np.percentile(selected, 1, axis=0)
    high = np.percentile(selected, 99, axis=0)
    extent = np.maximum(high - low, np.asarray([64.0, 128.0]))
    center = (low + high) * 0.5
    extent *= 1.25
    return np.asarray(
        [center[0] - extent[0] / 2, center[1] - extent[1] / 2, *extent],
        dtype=np.float32,
    )


def _decode_frames(
    video_path: Path,
    indices: np.ndarray,
    keypoints: np.ndarray,
    process_bbox,
    generate_patch_image,
    input_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    patches, boxes = [], []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise IOError(f"Cannot decode frame {index} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        bbox = process_bbox(
            _bbox_from_track(keypoints[int(index)], width, height), width, height
        )
        if bbox is None:
            capture.release()
            raise ValueError(f"Invalid signer crop at frame {index}: {video_path}")
        patch, _, _ = generate_patch_image(frame, bbox, 1.0, 0.0, False, input_shape)
        patches.append(patch.transpose(2, 0, 1) / 255.0)
        boxes.append(bbox)
    capture.release()
    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(boxes, dtype=np.float32),
        np.asarray([width, height], dtype=np.int32),
        fps,
    )


def _infer(
    model, patches: np.ndarray, batch_size: int, amp: bool
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {key: [] for key in TARGET_KEYS}
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            images = torch.from_numpy(patches[start : start + batch_size]).cuda(
                non_blocking=True
            )
            with torch.cuda.amp.autocast(enabled=amp):
                output = model({"img": images}, {}, {}, "test")
            for destination, source in TARGET_KEYS.items():
                value = output[source].detach().float().cpu().numpy()
                collected[destination].append(value)
    result = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite teacher output: {key}")
    return result


def _existing_valid(path: Path, frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return payload["body_pose"].shape == (frames, 63)
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/home/shared_data/sign_language/How2Sign")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--max-clips", type=int, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--clips-per-batch",
        type=int,
        default=1,
        help="Decode this many clips before a batched teacher call.",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=1,
        help="Parallel in-memory video decoders; use 5 for roughly 500%% CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_per_clip < 16:
        raise ValueError("Phase 2 sequences require at least 16 ordered frames")
    if (
        args.max_clips <= 0
        or args.batch_size <= 0
        or args.clips_per_batch <= 0
        or args.decode_workers <= 0
    ):
        raise ValueError(
            "max-clips, batch-size, clips-per-batch, and decode-workers must be positive"
        )
    if args.amp:
        raise ValueError(
            "SMPLer-X H32's installed MMCV ROIAlign requires FP32; AMP is unsafe"
        )
    root = args.root.resolve()
    output = args.output.resolve()
    video_dir, pose_dir = _split_paths(root, args.split)
    selected = _selection(
        video_dir, pose_dir, args.max_clips, args.frames_per_clip, args.seed
    )
    if len(selected) < args.max_clips:
        raise ValueError(
            f"Only {len(selected)} eligible clips; requested {args.max_clips}"
        )
    selection_payload = {
        "dataset": "How2Sign",
        "official_split": args.split,
        "motion_domain": "sign_language_asl",
        "teacher": MODEL_NAME,
        "frames_per_clip": args.frames_per_clip,
        "seed": args.seed,
        "clips": selected,
    }
    _write_selection(output / args.split / "selection.json", selection_payload)
    clip_dir = output / args.split / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        item
        for item in selected
        if not _existing_valid(
            clip_dir / f"{item['clip_id']}.npz", args.frames_per_clip
        )
    ]
    print(
        f"[selection] split={args.split} selected={len(selected)} "
        f"complete={len(selected) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    model, cfg = _load_model(output)
    from utils.preprocessing import generate_patch_image, process_bbox

    started = time.time()
    completed = len(selected) - len(pending)
    failures_path = output / args.split / "failures.jsonl"
    failed = 0

    def prepare(item: dict) -> dict:
        pose_path = Path(item["pose"])
        video_path = Path(item["video"])
        keypoints, scores = _load_track(pose_path)
        indices = np.linspace(
            0, len(keypoints) - 1, args.frames_per_clip, dtype=np.int64
        )
        patches, bboxes, image_size, fps = _decode_frames(
            video_path,
            indices,
            keypoints,
            process_bbox,
            generate_patch_image,
            cfg.input_img_shape,
        )
        return {
            "item": item,
            "patches": patches,
            "indices": indices,
            "keypoints": keypoints,
            "scores": scores,
            "bboxes": bboxes,
            "image_size": image_size,
            "fps": fps,
        }

    def submit_group(executor: ThreadPoolExecutor, group: list[dict]):
        return [(item, executor.submit(prepare, item)) for item in group]

    def resolve_group(futures) -> list[dict]:
        nonlocal failed
        decoded = []
        for item, future in futures:
            try:
                decoded.append(future.result())
            except Exception as error:
                failed += 1
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"clip_id": item["clip_id"], "error": repr(error)},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                print(f"[failure] clip={item['clip_id']} error={error!r}", flush=True)
        return decoded

    # Keep exactly one group prefetched.  Its five CPU decoders run while the
    # current group is on the GPU, avoiding the serial decode/infer bubbles
    # without retaining an unbounded number of full-resolution clips in RAM.
    groups = [
        pending[start : start + args.clips_per_batch]
        for start in range(0, len(pending), args.clips_per_batch)
    ]
    cv2.setNumThreads(1)
    executor = ThreadPoolExecutor(max_workers=args.decode_workers)
    current_futures = submit_group(executor, groups[0])
    try:
        for group_index in range(len(groups)):
            decoded = resolve_group(current_futures)
            next_futures = (
                submit_group(executor, groups[group_index + 1])
                if group_index + 1 < len(groups)
                else None
            )
            if decoded:
                patches = np.concatenate(
                    [entry["patches"] for entry in decoded], axis=0
                )
                prediction = _infer(model, patches, args.batch_size, args.amp)
                offset = 0
                for entry in decoded:
                    item = entry["item"]
                    end = offset + args.frames_per_clip
                    payload = {
                        **{key: value[offset:end] for key, value in prediction.items()},
                        "sample_indices": entry["indices"],
                        "keypoints_2d": entry["keypoints"][entry["indices"]],
                        "keypoint_scores": entry["scores"][entry["indices"]],
                        "bboxes": entry["bboxes"],
                        "image_size": entry["image_size"],
                        "fps": np.asarray(entry["fps"], dtype=np.float32),
                    }
                    destination = clip_dir / f"{item['clip_id']}.npz"
                    temporary = destination.with_suffix(".npz.partial")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **payload)
                    os.replace(temporary, destination)
                    offset = end
                    completed += 1
                if completed % 10 < len(decoded) or completed == len(selected):
                    elapsed = time.time() - started
                    rate = (completed - (len(selected) - len(pending))) / max(
                        elapsed, 1e-6
                    )
                    remaining = (len(selected) - completed) / max(rate, 1e-6)
                    print(
                        f"[progress] split={args.split} "
                        f"clips={completed}/{len(selected)} failed={failed} "
                        f"rate={rate:.3f}_clips/s "
                        f"eta_hours={remaining / 3600:.2f} "
                        "gpu_peak_gib="
                        f"{torch.cuda.max_memory_allocated() / 2**30:.2f}",
                        flush=True,
                    )
            current_futures = next_futures
    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
