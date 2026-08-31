"""Cache V1 hand candidates on signer-disjoint external validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase2_refiner.infer import _predict_sequence
from phase2_refiner.provenance import sha256_file
from signal4d_external.infer import _load_model
from signal4d_external.hand_v2_core import HAND_REGIONS, hand_eligibility


SPLITS = ("validation", "calibration")
FORBIDDEN_PARTS = {"sgnify", "smplx_gt", "evaluation_from_author"}


def _load_manifest(path: Path, expected_split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != expected_split:
        raise ValueError(
            f"Expected {expected_split} manifest, got {payload.get('split')}: {path}"
        )
    if int(payload.get("sgnify_training_reads", -1)) != 0:
        raise ValueError(f"Manifest does not prove zero SGNify reads: {path}")
    entries = payload.get("clips")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no clips: {path}")
    for item in entries:
        required = {"cache_path", "cache_sha256", "clip_id", "signer", "source_group"}
        if not required.issubset(item):
            raise ValueError(f"Malformed external manifest entry: {item}")
        cache_path = Path(item["cache_path"]).resolve()
        lowered = {part.lower() for part in cache_path.parts}
        if lowered & FORBIDDEN_PARTS:
            raise ValueError(f"Forbidden cache path: {cache_path}")
    return payload


def _activity_window(clip: Any, max_frames: int) -> slice:
    """Choose the most hand-active contiguous window without reading targets."""
    length = len(clip.frame_names)
    if length <= max_frames:
        return slice(0, length)
    activity = np.asarray(clip.hand_activity, dtype=np.float64).sum(axis=-1)
    totals = np.convolve(activity, np.ones(max_frames), mode="valid")
    start = int(np.argmax(totals))
    return slice(start, start + max_frames)


def _metadata(args: argparse.Namespace, manifests: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "SIGNAL4D_EXTERNAL_HAND_V2_PREDICTION_CACHE",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifests": {
            split: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for split, path in manifests.items()
        },
        "window_policy": "highest observable hand_activity contiguous window",
        "sgnify_training_or_selection_reads": 0,
    }


def _prepare_root(root: Path, metadata: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_metadata.json"
    rendered = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Prediction-cache provenance mismatch: {path}")
    else:
        path.write_text(rendered, encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {
        "validation": args.validation_manifest.resolve(),
        "calibration": args.calibration_manifest.resolve(),
    }
    payloads = {
        split: _load_manifest(path, split) for split, path in manifests.items()
    }
    overlap = set(payloads["validation"]["signers"]) & set(
        payloads["calibration"]["signers"]
    )
    if overlap:
        raise ValueError(f"Signer leakage between validation and calibration: {overlap}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    device = torch.device(args.device)
    model = _load_model(config, args.checkpoint.resolve(), device)
    max_frames = int(config["model"].get("max_frames", 64))
    reprojection_scale = float(
        config["data"].get("reprojection_residual_scale", 10.0)
    )
    metadata = _metadata(args, manifests)
    _prepare_root(args.output.resolve(), metadata)

    rows = []
    total = sum(len(payload["clips"]) for payload in payloads.values())
    done = 0
    for split in SPLITS:
        split_root = args.output.resolve() / split
        split_root.mkdir(exist_ok=True)
        for item in payloads[split]["clips"]:
            done += 1
            cache_path = Path(item["cache_path"]).resolve()
            if sha256_file(cache_path) != item["cache_sha256"]:
                raise ValueError(f"External cache hash mismatch: {cache_path}")
            output = split_root / f"{item['clip_id']}.npz"
            if not output.exists():
                clip = load_cache_clip(cache_path)
                clip_metadata = json.loads(clip.metadata_json)
                if clip_metadata.get("dataset") != "How2Sign":
                    raise ValueError(f"Non-How2Sign cache in V2H: {cache_path}")
                if int(clip_metadata.get("sgnify_training_reads", 0)) != 0:
                    raise ValueError(f"Cache reports SGNify training reads: {cache_path}")
                if clip.target_axis_angle is None:
                    raise ValueError(f"External cache has no target: {cache_path}")
                window = _activity_window(clip, max_frames)
                features, initial = features_from_clip(
                    clip,
                    window,
                    input_dim=45,
                    reprojection_residual_scale=reprojection_scale,
                )
                prediction = _predict_sequence(
                    model,
                    features,
                    initial,
                    torch.from_numpy(clip.refine_mask),
                    device,
                )
                target = axis_angle_to_matrix(
                    torch.from_numpy(clip.target_axis_angle[window]).float()
                )
                start = int(window.start or 0)
                stop = int(window.stop or len(clip.frame_names))
                temporary = output.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    clip_id=np.asarray(str(item["clip_id"])),
                    signer=np.asarray(str(item["signer"])),
                    source_group=np.asarray(str(item["source_group"])),
                    cache_path=np.asarray(str(cache_path)),
                    cache_sha256=np.asarray(str(item["cache_sha256"])),
                    frame_indices=np.arange(start, stop, dtype=np.int64),
                    timestamps=clip.timestamps[window],
                    initial_matrix=initial.cpu().numpy(),
                    candidate_matrix=prediction["matrix"].cpu().numpy(),
                    target_matrix=target.cpu().numpy(),
                    target_valid=clip.target_rotation_valid[window],
                    benefit_probability=prediction["benefit_logit"].sigmoid().cpu().numpy(),
                    **{
                        f"eligible_{region}": hand_eligibility(clip, region)[window]
                        for region in HAND_REGIONS
                    },
                )
                temporary.replace(output)
                status = "computed"
            else:
                status = "existing"
            rows.append(
                {
                    "split": split,
                    "clip_id": str(item["clip_id"]),
                    "signer": str(item["signer"]),
                    "source_group": str(item["source_group"]),
                    "output": str(output.resolve()),
                    "output_sha256": sha256_file(output),
                    "status": status,
                }
            )
            print(f"[hand-v2-predict] {done}/{total} {split} {item['clip_id']} {status}")

    index = {
        **metadata,
        "clips": rows,
        "clip_count": len(rows),
        "signers": {
            split: sorted(set(payloads[split]["signers"])) for split in SPLITS
        },
    }
    index_path = args.output.resolve() / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
