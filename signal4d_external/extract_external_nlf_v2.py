"""Extract NLF observations from preregistered external How2Sign video frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch

from phase2_refiner.provenance import sha256_file
from signal4d_v7_nlf_fusion.extract_nlf_observations import (
    detection_index,
    source_commit,
    write_detection,
)


def _batches(values: list[tuple[int, str]], size: int) -> Iterable[list[tuple[int, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _read_frames(video_path: Path, rows: list[tuple[int, str]]) -> list[torch.Tensor]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    images = []
    try:
        for frame_id, _ in rows:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Cannot decode {video_path}#frame={frame_id}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    finally:
        capture.release()
    return images


def _load_clips(manifests: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    clips = []
    provenance = []
    seen: set[tuple[str, int]] = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "signal4d.external_nlf_v2_manifest.v1":
            raise ValueError(f"Unsupported manifest: {path}")
        if int(payload.get("sgnify_training_reads", -1)) != 0:
            raise ValueError(f"Manifest does not prove zero SGNify reads: {path}")
        for clip in payload["clips"]:
            for frame_id in clip["frame_ids"]:
                key = (str(clip["clip_id"]), int(frame_id))
                if key in seen:
                    raise ValueError(f"Duplicate external frame: {key}")
                seen.add(key)
            clips.append(clip)
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
    return clips, provenance


def _existing_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(str(row["clip_id"]), int(row["frame_id"]))] = row
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    # The pinned NLF 0.3.2 TorchScript detector contains CUDA anchor constants.
    # Reject CPU explicitly instead of failing later with a mixed-device trace.
    if not str(args.device).startswith("cuda"):
        raise ValueError("Pinned NLF 0.3.2 requires a CUDA device")
    manifests = [path.resolve() for path in args.manifest]
    clips, manifest_provenance = _load_clips(manifests)
    if args.limit:
        remaining = args.limit
        limited = []
        for clip in clips:
            if remaining <= 0:
                break
            copy = dict(clip)
            copy["frame_ids"] = copy["frame_ids"][:remaining]
            copy["frame_names"] = copy["frame_names"][:remaining]
            remaining -= len(copy["frame_ids"])
            limited.append(copy)
        clips = limited

    model_path = args.model.resolve()
    expected = {
        "schema_version": "signal4d.external_nlf_v2_observations.v1",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "nlf_source_root": str(args.nlf_root.resolve()),
        "nlf_source_commit": source_commit(args.nlf_root.resolve()),
        "manifests": manifest_provenance,
        "sgnify_training_reads": 0,
        "settings": {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_aug": args.num_aug,
            "detector_threshold": args.detector_threshold,
            "selection": "max_box_area_times_score",
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / "run_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError("Existing NLF output has different inputs/settings")
    else:
        metadata_path.write_text(
            json.dumps({**expected, "created_unix": time.time()}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    index_path = args.output_root / "index.jsonl"
    records = _existing_records(index_path)
    model = torch.jit.load(str(model_path), map_location=args.device).eval()
    started = time.time()
    processed = 0
    for clip in clips:
        frame_rows = list(zip(clip["frame_ids"], clip["frame_names"]))
        for batch in _batches(frame_rows, args.batch_size):
            pending = [row for row in batch if (str(clip["clip_id"]), int(row[0])) not in records]
            if not pending:
                processed += len(batch)
                continue
            images = _read_frames(Path(clip["video_path"]), pending)
            shapes = {tuple(image.shape) for image in images}
            if len(shapes) != 1:
                raise ValueError(f"Mixed image shapes in {clip['clip_id']}: {sorted(shapes)}")
            tensor = torch.stack(images).to(args.device)
            with torch.inference_mode():
                prediction = model.detect_smpl_batched(
                    tensor,
                    model_name="smplx",
                    num_aug=args.num_aug,
                    detector_threshold=args.detector_threshold,
                )
            batch_records = []
            for image_index, (frame_id, frame_name) in enumerate(pending):
                output_relpath = (
                    Path(str(clip["split"]))
                    / str(clip["clip_id"])
                    / f"{int(frame_id):06d}.npz"
                )
                base = {
                    "clip_id": str(clip["clip_id"]),
                    "frame_id": int(frame_id),
                    "frame_name": str(frame_name),
                    "split": str(clip["split"]),
                    "signer": str(clip["signer"]),
                    "source_group": str(clip["source_group"]),
                    "cache_path": str(clip["cache_path"]),
                    "video_path": str(clip["video_path"]),
                    "output_relpath": str(output_relpath),
                }
                boxes = prediction["boxes"][image_index]
                if boxes.shape[0] == 0:
                    record = {**base, "status": "no_detection"}
                else:
                    person = detection_index(boxes)
                    stats = write_detection(
                        args.output_root / output_relpath,
                        prediction,
                        image_index,
                        person,
                    )
                    record = {**base, "status": "ok", **stats}
                records[(str(clip["clip_id"]), int(frame_id))] = record
                batch_records.append(record)
            with index_path.open("a", encoding="utf-8") as handle:
                for record in batch_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            processed += len(batch)
            print(
                f"[external-nlf-v2] {processed} frames elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    # Canonicalize once after the resumable append journal completes.
    rendered = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in sorted(
            records.values(),
            key=lambda value: (value["split"], value["clip_id"], value["frame_id"]),
        )
    )
    index_path.write_text(rendered, encoding="utf-8")
    failures = sum(row["status"] == "no_detection" for row in records.values())
    return {
        "frames": len(records),
        "no_detection": failures,
        "coverage": 1.0 - failures / max(len(records), 1),
        "index": str(index_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--nlf-root", type=Path, default=Path("nlf"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-aug", type=int, default=1)
    parser.add_argument("--detector-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_aug < 1:
        raise ValueError("batch-size and num-aug must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
