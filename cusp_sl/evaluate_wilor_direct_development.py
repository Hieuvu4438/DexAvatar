"""Evaluate WiLoR local-finger coordinate conversion on development targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from cusp_sl.evaluate_development import clustered_delta_interval
from cusp_sl.geometry import axis_angle_to_matrix, geodesic_distance
from cusp_sl.retargeting import fuse_wilor_hand, mano_fingers_to_smplx
from cusp_sl.temporal_filter import centered_tangent_filter, changed_joint_support
from cusp_sl.wilor_artifact import validate_wilor_raw_v3
from phase2_refiner.data.cache_schema import load_cache_clip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_detection(hands: list[dict], *, is_right: bool) -> dict | None:
    matching = [
        hand
        for hand in hands
        if bool(round(float(hand["is_right"]))) is is_right
    ]
    if not matching:
        return None
    if any("detector_confidence" not in hand for hand in matching):
        raise ValueError("WiLoR record lacks v3 detector_confidence")
    return max(matching, key=lambda hand: float(hand["detector_confidence"]))


def weighted(records: list[dict], name: str, weight_name: str) -> float:
    weights = np.asarray([record[weight_name] for record in records], dtype=np.float64)
    return float(np.average([record[name] for record in records], weights=weights))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wilor-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometric-wrist-alignment", action="store_true")
    parser.add_argument("--temporal-filter-radius", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    args.output.mkdir(parents=True)

    # The pickle is a locally generated, hash-recorded WiLoR artifact. Never
    # accept an untrusted downloaded pickle on this path.
    with args.wilor_pickle.open("rb") as handle:
        wilor = pickle.load(handle, encoding="latin1")
    images, wilor_meta = validate_wilor_raw_v3(wilor)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records: list[dict] = []
    for entry in manifest["clips"]:
        path = Path(entry) if Path(entry).is_absolute() else args.manifest.parent / entry
        clip = load_cache_clip(path)
        if clip.target_axis_angle is None or clip.target_rotation_valid is None:
            raise ValueError(f"Development clip lacks rotation target: {path}")
        metadata = json.loads(clip.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Development clip lacks source_group: {path}")

        base = axis_angle_to_matrix(torch.from_numpy(clip.init_axis_angle).float())
        global_orient = axis_angle_to_matrix(
            torch.from_numpy(clip.global_orient).float()
        )
        target = axis_angle_to_matrix(torch.from_numpy(clip.target_axis_angle).float())
        candidate = base.clone()
        detected = {"left": 0, "right": 0}
        for frame, frame_name in enumerate(clip.frame_names.astype(str)):
            image_record = images.get(f"{frame_name}.png")
            if image_record is None:
                raise ValueError(f"Missing WiLoR image record: {frame_name}.png")
            hands = image_record.get("hands", [])
            for side, is_right, start in (
                ("left", False, 21),
                ("right", True, 36),
            ):
                hand = select_detection(hands, is_right=is_right)
                if hand is None:
                    continue
                rotation = torch.from_numpy(
                    np.asarray(hand["pred_mano_pose_rotmat"], dtype=np.float32)
                )
                if args.geometric_wrist_alignment:
                    wrist = torch.from_numpy(
                        np.asarray(
                            hand["pred_mano_global_orient_rotmat"], dtype=np.float32
                        )
                    ).reshape(3, 3)
                    candidate[frame] = fuse_wilor_hand(
                        candidate[frame],
                        global_orient[frame],
                        rotation,
                        wrist,
                        is_right=is_right,
                    )
                else:
                    candidate[frame, start : start + 15] = mano_fingers_to_smplx(
                        rotation, is_right=is_right
                    )
                detected[side] += 1

        if args.temporal_filter_radius < 0:
            raise ValueError("--temporal-filter-radius must be non-negative")
        if args.temporal_filter_radius:
            filter_support = changed_joint_support(base, candidate)
            filter_support &= torch.from_numpy(clip.refine_mask).bool()
            candidate = centered_tangent_filter(
                candidate,
                filter_support,
                radius=args.temporal_filter_radius,
            )

        valid = torch.from_numpy(clip.target_rotation_valid).bool()
        valid &= torch.from_numpy(clip.refine_mask)[None].bool()
        error_base = torch.rad2deg(geodesic_distance(base, target))
        error_candidate = torch.rad2deg(geodesic_distance(candidate, target))
        joint_index = torch.arange(51)[None]
        masks = {
            "overall": valid,
            "body": valid & (joint_index < 21),
            "hands": valid & (joint_index >= 21),
            "left": valid & ((joint_index >= 21) & (joint_index < 36)),
            "right": valid & (joint_index >= 36),
        }
        record: dict[str, object] = {
            "clip_id": clip.clip_id,
            "source_group": source_group,
            "left_detected_frames": detected["left"],
            "right_detected_frames": detected["right"],
            "frames": len(clip.frame_names),
        }
        for group, mask in masks.items():
            tokens = int(mask.sum())
            record[f"{group}_tokens"] = tokens
            record[f"base_{group}_degrees"] = float(error_base[mask].mean())
            record[f"direct_{group}_degrees"] = float(error_candidate[mask].mean())
        # Generic names let the shared cluster-bootstrap implementation operate
        # on the primary overall endpoint.
        record["tokens"] = record["overall_tokens"]
        record["base_degrees"] = record["base_overall_degrees"]
        record["direct_degrees"] = record["direct_overall_degrees"]
        records.append(record)

    evaluated_frames = int(sum(record["frames"] for record in records))
    if evaluated_frames != len(images):
        raise ValueError(
            f"WiLoR artifact has {len(images)} records for "
            f"{evaluated_frames} development frames"
        )

    csv_path = args.output / "per_clip.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary: dict[str, object] = {
        "role": "development_only",
        "baseline": (
            "smpler_x_plus_wilor_geometric_wrist_alignment"
            if args.geometric_wrist_alignment
            else "smpler_x_plus_wilor_coordinate_conversion_only"
        ),
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
        "clips": len(records),
        "frames": int(sum(record["frames"] for record in records)),
        "left_detection_frame_fraction": float(
            sum(record["left_detected_frames"] for record in records)
            / sum(record["frames"] for record in records)
        ),
        "right_detection_frame_fraction": float(
            sum(record["right_detected_frames"] for record in records)
            / sum(record["frames"] for record in records)
        ),
        "manifest_sha256": sha256(args.manifest),
        "wilor_pickle_sha256": sha256(args.wilor_pickle),
        "wilor_frame_manifest_sha256": wilor_meta["frame_manifest_sha256"],
        "wilor_checkpoint_sha256": wilor_meta["wilor_checkpoint_sha256"],
        "wilor_detector_checkpoint_sha256": wilor_meta[
            "detector_checkpoint_sha256"
        ],
    }
    for group in ("overall", "body", "hands", "left", "right"):
        summary[f"base_{group}_degrees"] = weighted(
            records, f"base_{group}_degrees", f"{group}_tokens"
        )
        summary[f"direct_{group}_degrees"] = weighted(
            records, f"direct_{group}_degrees", f"{group}_tokens"
        )
    summary["clustered_direct_minus_base"] = clustered_delta_interval(
        records,
        "direct_degrees",
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary["clustered_direct_minus_base_by_group"] = {
        group: clustered_delta_interval(
            records,
            f"direct_{group}_degrees",
            replicates=args.bootstrap_replicates,
            seed=args.seed + 100 + index,
            weight_key=f"{group}_tokens",
            base_key=f"base_{group}_degrees",
        )
        for index, group in enumerate(("body", "hands", "left", "right"))
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
