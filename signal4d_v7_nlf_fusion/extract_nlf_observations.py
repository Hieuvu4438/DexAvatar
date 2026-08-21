#!/usr/bin/env python3
"""Export uncertainty-bearing NLF observations on a frozen SIGNAL-4D manifest.

The historical NLF adapter reduced NLF to SMPL-X parameters and discarded the
non-parametric localizer-field predictions and uncertainties. This exporter
preserves those signals for an isolated, uncertainty-weighted fusion lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(nlf_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(nlf_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_manifest_frames(manifests: Sequence[Path], data_root: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                clip = json.loads(line)
                frame_ids = clip["frame_ids"]
                image_relpaths = clip["image_relpaths"]
                if len(frame_ids) != len(image_relpaths):
                    raise ValueError(
                        f"{manifest}:{line_number}: frame_ids/image_relpaths length mismatch"
                    )
                for frame_id, image_relpath in zip(frame_ids, image_relpaths):
                    key = (str(clip["clip_id"]), int(frame_id))
                    if key in seen:
                        raise ValueError(f"duplicate manifest frame {key}")
                    seen.add(key)
                    image_path = data_root / image_relpath
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    frames.append(
                        {
                            "clip_id": key[0],
                            "frame_id": key[1],
                            "split": str(clip["split"]),
                            "image_relpath": str(image_relpath),
                            "image_path": image_path,
                        }
                    )
    return frames


def detection_index(boxes: torch.Tensor) -> int:
    """Select the principal signer using detector geometry and confidence.

    NLF boxes follow ``[x, y, width, height, score]``. Vertex spread is not a
    valid person-size proxy because parametric bodies have similar canonical
    extent regardless of their image-space scale.
    """
    if boxes.ndim != 2 or boxes.shape[1] < 4 or boxes.shape[0] == 0:
        raise ValueError(f"invalid/empty NLF boxes with shape {tuple(boxes.shape)}")
    areas = boxes[:, 2].clamp_min(0) * boxes[:, 3].clamp_min(0)
    if boxes.shape[1] >= 5:
        scores = boxes[:, 4].clamp_min(0)
        rank = areas * scores.clamp_min(1e-6)
    else:
        rank = areas
    return int(torch.argmax(rank).item())


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def write_detection(
    output_path: Path,
    prediction: Dict[str, Any],
    image_index: int,
    person_index: int,
) -> Dict[str, Any]:
    arrays: Dict[str, np.ndarray] = {}
    keys = (
        "boxes",
        "pose",
        "betas",
        "trans",
        "vertices3d",
        "joints3d",
        "vertices2d",
        "joints2d",
        "vertices3d_nonparam",
        "joints3d_nonparam",
        "vertices2d_nonparam",
        "joints2d_nonparam",
        "vertex_uncertainties",
        "joint_uncertainties",
    )
    for key in keys:
        value = prediction[key][image_index][person_index]
        arrays[key] = tensor_numpy(value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "detection_count": int(prediction["boxes"][image_index].shape[0]),
        "selected_detection": person_index,
        "box": arrays["boxes"].tolist(),
        "mean_vertex_uncertainty_mm": float(arrays["vertex_uncertainties"].mean()),
        "mean_joint_uncertainty_mm": float(arrays["joint_uncertainties"].mean()),
    }


def batches(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--nlf-root", type=Path, default=Path("nlf"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-aug", type=int, default=1)
    parser.add_argument("--detector-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = [path.resolve() for path in args.manifest]
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.batch_size < 1 or args.num_aug < 1:
        raise ValueError("batch-size and num-aug must be positive")

    frames = load_manifest_frames(manifests, args.data_root.resolve())
    if args.limit:
        frames = frames[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": "signal4d.nlf_observations.v1",
        "created_unix": time.time(),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "nlf_source_root": str(args.nlf_root.resolve()),
        "nlf_source_commit": source_commit(args.nlf_root.resolve()),
        "manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in manifests
        ],
        "frame_count": len(frames),
        "settings": {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_aug": args.num_aug,
            "detector_threshold": args.detector_threshold,
            "model_name": "smplx",
            "selection": "max_box_area_times_score",
        },
    }
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _ = torchvision.io
    model = torch.jit.load(str(model_path), map_location=args.device).eval()
    records: List[Dict[str, Any]] = []
    started = time.time()

    for batch_number, frame_batch in enumerate(batches(frames, args.batch_size), start=1):
        pending: List[Dict[str, Any]] = []
        for frame in frame_batch:
            output_relpath = Path(frame["split"]) / frame["clip_id"] / f"{frame['frame_id']:06d}.npz"
            output_path = args.output_root / output_relpath
            if output_path.is_file() and not args.overwrite:
                records.append({**frame, "output_relpath": str(output_relpath), "status": "existing"})
            else:
                pending.append({**frame, "output_relpath": output_relpath, "output_path": output_path})
        if not pending:
            continue

        images = [torchvision.io.read_image(str(frame["image_path"])) for frame in pending]
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(f"mixed image shapes in batch: {sorted(shapes)}")
        image_tensor = torch.stack(images).to(args.device)
        with torch.inference_mode():
            prediction = model.detect_smpl_batched(
                image_tensor,
                model_name="smplx",
                num_aug=args.num_aug,
                detector_threshold=args.detector_threshold,
            )

        for image_index, frame in enumerate(pending):
            boxes_i = prediction["boxes"][image_index]
            base_record = {
                "clip_id": frame["clip_id"],
                "frame_id": frame["frame_id"],
                "split": frame["split"],
                "image_relpath": frame["image_relpath"],
                "output_relpath": str(frame["output_relpath"]),
            }
            if boxes_i.shape[0] == 0:
                records.append({**base_record, "status": "no_detection"})
                continue
            person_index = detection_index(boxes_i)
            stats = write_detection(frame["output_path"], prediction, image_index, person_index)
            records.append({**base_record, "status": "ok", **stats})

        processed = min(batch_number * args.batch_size, len(frames))
        print(f"[NLF-V7] {processed}/{len(frames)} elapsed={time.time() - started:.1f}s", flush=True)

    records.sort(key=lambda row: (row["split"], row["clip_id"], row["frame_id"]))
    index_path = args.output_root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            record.pop("image_path", None)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    failures = sum(record["status"] == "no_detection" for record in records)
    print(
        f"[NLF-V7] complete frames={len(records)} no_detection={failures} "
        f"index={index_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
