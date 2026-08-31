"""Materialize immutable strong-A1 caches from SMPLer-X templates and WiLoR v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cusp_sl.evaluate_wilor_direct_development import select_detection
from cusp_sl.geometry import (
    axis_angle_to_matrix,
    geodesic_distance,
    matrix_to_axis_angle,
)
from cusp_sl.prepare_wilor_predictions import wilor_candidate
from cusp_sl.temporal_filter import centered_tangent_filter, changed_joint_support
from cusp_sl.wilor_artifact import validate_wilor_raw_v3
from phase2_refiner.data.add_reprojection_residuals import (
    _how2sign_batch,
    _lane_clip,
)
from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip
from phase2_refiner.render import create_smplx_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hand_observation_metadata(
    hands: list[dict], *, is_right: bool, width: int, height: int
) -> tuple[dict | None, float, float, bool]:
    matching = [
        hand
        for hand in hands
        if bool(round(float(hand["is_right"]))) is is_right
    ]
    hand = select_detection(hands, is_right=is_right)
    if hand is None:
        return None, 0.0, 1.0, False
    size = float(hand["box_size"])
    center_x, center_y = np.asarray(hand["box_center"], dtype=np.float32)
    scale = size / max(float(max(width, height)), 1.0)
    half = size * 0.5
    outside = max(0.0, half - center_x) + max(0.0, center_x + half - width)
    outside += max(0.0, half - center_y) + max(0.0, center_y + half - height)
    truncation = min(1.0, outside / max(2.0 * size, 1.0))
    return hand, scale, truncation, len(matching) > 1


def _frame_records(clip, images: dict[str, dict]) -> list[dict]:
    result = []
    for name in clip.frame_names.astype(str):
        keys = (f"{clip.clip_id}__{name}.png", f"{name}.png")
        record = next((images[key] for key in keys if key in images), None)
        if record is None:
            raise ValueError(f"Missing WiLoR record for {clip.clip_id}/{name}")
        result.append(record)
    return result


def _update_contract(
    clip,
    image_records: list[dict],
    manifest_records: dict[str, dict],
    selected_axis_angle: np.ndarray,
    *,
    geometric_wrist_alignment: bool,
    wilor_meta: dict,
):
    observations = clip.observation_features.copy()
    base_rotation = axis_angle_to_matrix(torch.from_numpy(clip.init_axis_angle).float())
    selected_rotation = axis_angle_to_matrix(
        torch.from_numpy(selected_axis_angle).float()
    )
    disagreement = (
        geodesic_distance(base_rotation, selected_rotation).numpy() / np.pi
    )
    observations[..., 7] = disagreement
    component = []
    fallback = []
    for frame, (name, image_record) in enumerate(
        zip(clip.frame_names.astype(str), image_records, strict=True)
    ):
        keys = (f"{clip.clip_id}__{name}.png", f"{name}.png")
        manifest_record = next(
            (manifest_records[key] for key in keys if key in manifest_records), None
        )
        if manifest_record is None:
            raise ValueError(f"Frame manifest lacks {clip.clip_id}/{name}")
        width = int(manifest_record["expected_width"])
        height = int(manifest_record["expected_height"])
        detected_sides = []
        missing_sides = []
        hands = image_record.get("hands", [])
        for side, is_right, start in (
            ("left", False, 21),
            ("right", True, 36),
        ):
            stop = start + 15
            hand, crop_scale, truncation, duplicate = hand_observation_metadata(
                hands, is_right=is_right, width=width, height=height
            )
            if hand is None:
                observations[frame, start:stop, 0] = 0.0
                observations[frame, start:stop, 1] = 0.0
                observations[frame, start:stop, 2] = 1.0
                observations[frame, start:stop, 3] = 0.0
                observations[frame, start:stop, 4] = 1.0
                observations[frame, start:stop, 6] = 0.0
                missing_sides.append(side)
                continue
            confidence = np.clip(float(hand["detector_confidence"]), 0.0, 1.0)
            observations[frame, start:stop, 0] = confidence
            observations[frame, start:stop, 1] = 1.0
            observations[frame, start:stop, 2] = 0.0
            observations[frame, start:stop, 3] = crop_scale
            observations[frame, start:stop, 4] = truncation
            observations[frame, start:stop, 6] = float(duplicate)
            detected_sides.append(side)
        component.append(
            "smplerx_plus_wilor_" + ("_".join(detected_sides) or "fallback")
        )
        fallback.append(
            "" if not missing_sides else "wilor_dropout_" + "_".join(missing_sides)
        )

    innovation = np.zeros_like(observations[..., 5])
    if len(selected_rotation) > 1:
        delta = (
            geodesic_distance(selected_rotation[1:], selected_rotation[:-1]).numpy()
            / np.pi
        )
        innovation[1:] = delta
        innovation[0] = delta[0]
    observations[..., 5] = np.clip(innovation, 0.0, 1.0)
    confidence = np.clip(observations[..., 0], 0.0, 1.0)
    presence = np.clip(observations[..., 1], 0.0, 1.0)
    missing = np.clip(observations[..., 2], 0.0, 1.0)
    truncation = np.clip(observations[..., 4], 0.0, 1.0)
    duplicate = np.clip(observations[..., 6], 0.0, 1.0)
    reliability = (
        confidence
        * presence
        * (1.0 - missing)
        * (1.0 - truncation)
        * (1.0 - 0.5 * duplicate)
        * np.exp(-2.0 * innovation)
    ).astype(np.float32)
    detector_present = observations[..., 1] > 0.5
    track_valid = clip.keypoint_valid & detector_present
    metadata = json.loads(clip.metadata_json)
    metadata.update(
        {
            "initializer_expert": "frozen SMPLer-X plus WiLoR raw-v3",
            "initializer_matches_locked_lane_a1": True,
            "wilor_frame_manifest": wilor_meta["frame_manifest"],
            "wilor_frame_manifest_sha256": wilor_meta["frame_manifest_sha256"],
            "wilor_checkpoint_sha256": wilor_meta["wilor_checkpoint_sha256"],
            "wilor_detector_checkpoint_sha256": wilor_meta[
                "detector_checkpoint_sha256"
            ],
            "wilor_confidence_calibration": "identity_unvalidated",
            "wilor_geometric_wrist_alignment": geometric_wrist_alignment,
            "requires_reprojection_enrichment": True,
        }
    )
    metadata.pop("reprojection_residual_provider", None)
    metadata.pop("reprojection_residual_clipping_fraction", None)
    return replace(
        clip,
        init_axis_angle=selected_axis_angle,
        observation_features=observations,
        u0_reliability=reliability,
        raw_confidence=confidence.astype(np.float32),
        calibrated_confidence=confidence.astype(np.float32),
        detector_present=detector_present,
        track_valid=track_valid,
        in_frame=observations[..., 4] <= 0.5,
        copied_observation=observations[..., 6] > 0.5,
        initializer_component=np.asarray(component, dtype=str),
        fallback_reason=np.asarray(fallback, dtype=str),
        hand_activity=np.stack(
            (track_valid[:, 21:36].mean(1), track_valid[:, 36:51].mean(1)),
            axis=-1,
        ).astype(np.float32),
        reprojection_residual_2d=np.zeros_like(clip.reprojection_residual_2d),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wilor-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("how2sign", "sgnify"), required=True)
    parser.add_argument("--smplx-model-folder", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometric-wrist-alignment", action="store_true")
    parser.add_argument("--temporal-filter-radius", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.temporal_filter_radius < 0:
        raise ValueError("--temporal-filter-radius must be non-negative")
    with args.wilor_pickle.open("rb") as handle:
        wilor = pickle.load(handle, encoding="latin1")
    images, wilor_meta = validate_wilor_raw_v3(wilor)
    frame_manifest = json.loads(
        Path(wilor_meta["frame_manifest"]).read_text(encoding="utf-8")
    )
    frame_records = {
        str(record["image_key"]): record for record in frame_manifest["records"]
    }
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = create_smplx_model(args.smplx_model_folder, device)
    model.requires_grad_(False)
    (args.output / "clips").mkdir(parents=True)
    entries = []
    summaries = []
    frames = 0
    for entry in source["clips"]:
        relative = entry["cache"] if isinstance(entry, dict) else entry
        path = Path(relative)
        if not path.is_absolute():
            path = args.manifest.parent / path
        clip = load_cache_clip(path)
        image_records = _frame_records(clip, images)
        selected, detected = wilor_candidate(
            clip.init_axis_angle,
            clip.global_orient,
            image_records,
            geometric_wrist_alignment=args.geometric_wrist_alignment,
        )
        if args.temporal_filter_radius:
            base_rotation = axis_angle_to_matrix(
                torch.from_numpy(clip.init_axis_angle).float()
            )
            selected_rotation = axis_angle_to_matrix(
                torch.from_numpy(selected).float()
            )
            filter_support = changed_joint_support(
                base_rotation, selected_rotation
            )
            filter_support &= torch.from_numpy(clip.refine_mask).bool()
            selected_rotation = centered_tangent_filter(
                selected_rotation,
                filter_support,
                radius=args.temporal_filter_radius,
            )
            selected = matrix_to_axis_angle(selected_rotation).numpy().astype(
                np.float32
            )
        updated = _update_contract(
            clip,
            image_records,
            frame_records,
            selected,
            geometric_wrist_alignment=args.geometric_wrist_alignment,
            wilor_meta=wilor_meta,
        )
        if args.mode == "how2sign":
            residual, clipping = _how2sign_batch(model, [updated], device)[0]
        else:
            residual, clipping = _lane_clip(model, updated, device)
        metadata = json.loads(updated.metadata_json)
        metadata["reprojection_residual_provider"] = (
            "frozen strong-A1 SMPL-X joint reprojection v1"
        )
        metadata["reprojection_residual_clipping_fraction"] = clipping
        metadata.pop("requires_reprojection_enrichment", None)
        updated = replace(
            updated,
            reprojection_residual_2d=residual,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        destination = args.output / "clips" / f"{clip.clip_id}.npz"
        save_cache_clip(destination, updated)
        entries.append(str(Path("clips") / destination.name))
        frames += len(clip.frame_names)
        summaries.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(clip.frame_names),
                "left_detected_frames": detected["left"],
                "right_detected_frames": detected["right"],
                "source_cache_sha256": sha256(path),
                "derived_cache_sha256": sha256(destination),
                "reprojection_clipping_fraction": clipping,
            }
        )
        print(f"[a1-cache] {clip.clip_id}: {len(clip.frame_names)} frames")
    expected = source.get("expected_frames")
    if expected is not None and frames != int(expected):
        raise ValueError(f"Derived {frames} != expected {expected}")
    if frames != len(images):
        raise ValueError(
            f"WiLoR artifact has {len(images)} records for {frames} source frames"
        )
    report = {
        "role": "frozen_strong_a1_derived_cache",
        "mode": args.mode,
        "geometric_wrist_alignment": args.geometric_wrist_alignment,
        "temporal_filter": (
            "centered_so3_tangent_triangular"
            if args.temporal_filter_radius
            else None
        ),
        "temporal_filter_radius": args.temporal_filter_radius,
        "temporal_filter_joint_support": (
            "a1_changed_joints_intersect_cache_refine_mask"
            if args.temporal_filter_radius
            else None
        ),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "wilor_pickle": str(args.wilor_pickle.resolve()),
        "wilor_pickle_sha256": sha256(args.wilor_pickle),
        "wilor_frame_manifest_sha256": wilor_meta["frame_manifest_sha256"],
        "clips": entries,
        "clip_count": len(entries),
        "expected_frames": frames,
        "summaries": summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"clips": len(entries), "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
