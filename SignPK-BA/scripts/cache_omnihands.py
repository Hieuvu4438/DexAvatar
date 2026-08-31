#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.data.frame_manifest import SignManifest
from signpk.data.window_sampler import all_windows
from signpk.observers.h4w_wrapper import load_h4w_cache
from signpk.observers.omnihands_wrapper import export_omnihands_output
from signpk.utils.config import load_yaml, project_path
from signpk.utils.config_hash import config_hash, sha256_file
from signpk.utils.reproducibility import set_deterministic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _stable_boxes(boxes: torch.Tensor, image_sizes: list[tuple[int, int]], radius: int = 2) -> torch.Tensor:
    result = boxes.clone()
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(4)
    for index in range(len(boxes)):
        lo, hi = max(0, index - radius), min(len(boxes), index + radius + 1)
        stable_size = sizes[lo:hi].median(0).values
        width, height = image_sizes[index][1], image_sizes[index][0]
        half = stable_size * 0.5
        result[index, :2] = torch.maximum(centers[index] - half, boxes.new_zeros(2))
        result[index, 2:] = torch.minimum(centers[index] + half, boxes.new_tensor([width, height]))
    return result


def _fallback_invalid_boxes(boxes: torch.Tensor, valid: torch.Tensor, image_sizes: list[tuple[int, int]]) -> torch.Tensor:
    result = boxes.clone()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    for index in range(len(result)):
        if valid[index]:
            continue
        if valid_indices.numel():
            nearest = valid_indices[torch.argmin(torch.abs(valid_indices - index))]
            result[index] = result[nearest]
        else:
            height, width = image_sizes[index]
            result[index] = result.new_tensor([0, 0, width, height])
    return result


def _load_images(manifest: SignManifest) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
    images, sizes = [], []
    for record in manifest.records:
        image = cv2.imread(str(record.rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record.rgb_path)
        images.append(torch.from_numpy(image.copy()))
        sizes.append(image.shape[:2])
    return images, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache OmniHands video observations before rendering")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data/sgnify.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sign", action="append", help="repeat to restrict signs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size-token", type=int, default=24)
    parser.add_argument("--batch-size-temporal", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    set_deterministic(42, deterministic=True)
    config = load_yaml(args.config)
    data_config, observer_config = config["data"], config["observers"]
    omni_config = observer_config["omnihands"]
    source_root = project_path(omni_config["source_root"], PROJECT_ROOT)
    checkpoint = project_path(omni_config["checkpoint"], PROJECT_ROOT)
    model_config_path = project_path(omni_config["config"], PROJECT_ROOT)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OmniHands checkpoint missing: {checkpoint}")
    revision = _git_revision(source_root)
    expected = omni_config.get("expected_commit")
    if expected and revision != expected:
        raise ValueError(f"OmniHands revision mismatch: {revision} != {expected}")
    sys.path.insert(0, str(source_root))
    with working_directory(source_root):
        from hands_4d.datasets.vitdet_dataset import (
            HandToken_Sequence,
            ViTDetInterDataset_Batch,
            ViTDetInterDataset_Sequence,
        )
        from hands_4d.models import load_from_ckpt
        from hands_4d.utils import recursive_to

        model, model_config = load_from_ckpt(str(checkpoint), str(model_config_path))
        device = torch.device(args.device)
        model = model.to(device).eval()
        window_size = int(omni_config.get("sequence_length", 9))
        gap = int(omni_config.get("sequence_gap", 1))
        if int(model_config.MODEL.SEQ_LEN) != window_size:
            raise ValueError(
                f"checkpoint config sequence length {model_config.MODEL.SEQ_LEN} != requested {window_size}"
            )
        output_root = args.output_root.resolve() if args.output_root else project_path(omni_config["cache_root"], PROJECT_ROOT)
        manifest_root = project_path(data_config["manifest_root"], PROJECT_ROOT)
        h4w_root = project_path(observer_config["h4w"]["cache_root"], PROJECT_ROOT)
        paths = sorted(manifest_root.glob("*/manifest.json"))
        if args.sign:
            selected = set(args.sign)
            paths = [path for path in paths if path.parent.name in selected]
        for manifest_path in paths:
            manifest = SignManifest.load(manifest_path, validate_paths=True)
            target = output_root / manifest.sign_name / "omni.pt"
            if target.exists() and not args.overwrite:
                print(f"[skip] {manifest.sign_name}: {target}")
                continue
            _, h4w_left, h4w_right, _ = load_h4w_cache(
                h4w_root, manifest, expected_commit=observer_config["h4w"].get("expected_commit")
            )
            images, image_sizes = _load_images(manifest)
            left_boxes = _stable_boxes(_fallback_invalid_boxes(h4w_left.bbox_xyxy, h4w_left.valid, image_sizes), image_sizes)
            right_boxes = _stable_boxes(_fallback_invalid_boxes(h4w_right.bbox_xyxy, h4w_right.valid, image_sizes), image_sizes)
            boxes = {
                "left": {str(i): left_boxes[i].tolist() for i in range(len(images))},
                "right": {str(i): right_boxes[i].tolist() for i in range(len(images))},
            }
            specs = all_windows(len(images), window_size, gap, data_config.get("boundary_padding", "reflect"))
            sequences = np.asarray([spec.indices for spec in specs], dtype=np.int64)
            token_dataset = ViTDetInterDataset_Batch(model_config, images, boxes, rescale_factor=2.0)
            token_loader = DataLoader(token_dataset, batch_size=args.batch_size_token, shuffle=False, num_workers=args.num_workers)
            tokens = []
            with torch.inference_mode():
                for batch in token_loader:
                    batch = recursive_to(batch, device)
                    values = model.inference_token_forward(batch).reshape(-1, 2, 1024)
                    tokens.append(values.cpu())
            all_tokens = torch.cat(tokens, dim=0)
            sequence_images = ViTDetInterDataset_Sequence(model_config, images, boxes, sequences, rescale_factor=2.0)
            sequence_tokens = HandToken_Sequence(all_tokens, sequences)
            image_loader = DataLoader(sequence_images, batch_size=args.batch_size_temporal, shuffle=False, num_workers=args.num_workers)
            sequence_token_loader = DataLoader(sequence_tokens, batch_size=args.batch_size_temporal, shuffle=False, num_workers=args.num_workers)
            output_rows: dict[str, list[torch.Tensor]] = {}
            with torch.inference_mode():
                for token_batch, image_batch in zip(sequence_token_loader, image_loader):
                    token_batch = token_batch.reshape(-1, 1024).to(device)
                    image_batch = recursive_to(image_batch, device)
                    output = model.inference_temp_forward(token_batch, image_batch)
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            output_rows.setdefault(key, []).append(value.detach().cpu())
            merged = {key: torch.cat(values, dim=0) for key, values in output_rows.items()}
            window_metadata = [
                {"center_index": spec.center_index, "indices": list(spec.indices), "padded": list(spec.padded)}
                for spec in specs
            ]
            export_omnihands_output(
                merged,
                all_tokens,
                manifest,
                window_metadata,
                {"left": left_boxes, "right": right_boxes},
                {"left": h4w_left.valid, "right": h4w_right.valid},
                target,
                {
                    "observer": "OmniHands-video",
                    "repository_commit": revision,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "config": str(model_config_path),
                    "config_hash": config_hash(config),
                    "sequence_length": window_size,
                    "sequence_gap": gap,
                    "boundary_padding": data_config.get("boundary_padding", "reflect"),
                    "units": "meters",
                    "coordinates": "native_omnihands; adapter_required",
                    "left_world_x_mirror_applied_upstream": True,
                    "local_joint_order": "wrist_thumb_index_middle_ring_pinky",
                    "local_wrist_index": 0,
                    "note": "raw world-relative outputs retain OmniHands upstream index-9 anchor",
                },
            )
            print(f"[done] {manifest.sign_name}: {len(images)} frames -> {target}")


if __name__ == "__main__":
    main()
