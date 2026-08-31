"""Attach released How2Sign-Synth3D SMPL-H targets to external caches.

The existing How2Sign cache contains the observation distribution used by the
frozen SIGNAL4D external lane (SMPLer-X initializer, 2D tracks, reliability,
and reprojection residuals).  This append-only adapter preserves those inputs
and replaces only the old 2D-guided pseudo-target with the released synchronized
SMPL-H pose for the exact source frames.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import CacheClip, load_cache_clip, save_cache_clip
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from phase2_refiner.provenance import sha256_file


NUM_JOINTS = 51
REGIONS = {"ubody": slice(0, 21), "lhand": slice(21, 36), "rhand": slice(36, 51)}
SPLITS = ("train", "val", "calibration")
FRONT_SUFFIX = "-rgb_front"
TARGET_PROVIDER = "How2Sign-Synth3D synchronized green-screen SMPL-H v1"


@dataclass(frozen=True)
class Annotation:
    sentence_name: str
    video_name: str
    start: float
    end: float
    aligned: bool

    @property
    def sequence_name(self) -> str:
        if not self.video_name.endswith(FRONT_SUFFIX):
            raise ValueError(f"Unexpected How2Sign VIDEO_NAME: {self.video_name!r}")
        return self.video_name[: -len(FRONT_SUFFIX)]


@dataclass(frozen=True)
class Candidate:
    split: str
    official_split: str
    source: Path
    annotation: Annotation
    annotation_path: Path
    fit_path: Path


def _load_annotations(path: Path) -> dict[str, Annotation]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, Annotation] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"VIDEO_NAME", "SENTENCE_NAME", "START", "END", "ALIGNED"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        for raw in reader:
            name = raw["SENTENCE_NAME"].strip()
            if not name:
                continue
            if name in rows:
                raise ValueError(f"Duplicate SENTENCE_NAME in {path}: {name}")
            rows[name] = Annotation(
                sentence_name=name,
                video_name=raw["VIDEO_NAME"].strip(),
                start=float(raw["START"]),
                end=float(raw["END"]),
                aligned=raw["ALIGNED"].strip().lower() == "true",
            )
    if not rows:
        raise ValueError(f"No annotations in {path}")
    return rows


def _nearest_frame(seconds: float, fps: float) -> int:
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {seconds}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    # Explicit half-up rounding avoids Python's banker-rounding ambiguity.
    return int(np.floor(seconds * fps + 0.5))


def target_frame_indices(clip: CacheClip, annotation: Annotation) -> np.ndarray:
    base = _nearest_frame(annotation.start, float(clip.fps))
    indices = base + clip.frame_numbers.astype(np.int64)
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"Invalid target frame binding for {clip.clip_id}")
    return indices


def _load_smplh_fit(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "name",
            "gender",
            "beta",
            "thetas",
            "translations",
            "camera_to_world",
            "camera_to_image",
            "camera_heights",
            "camera_widths",
            "camera_names",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} lacks arrays: {sorted(missing)}")
        result = {name: payload[name] for name in payload.files}
    theta = np.asarray(result["thetas"], dtype=np.float32)
    if theta.ndim != 3 or theta.shape[1:] != (52, 3) or len(theta) == 0:
        raise ValueError(f"Invalid SMPL-H theta shape in {path}: {theta.shape}")
    if not np.isfinite(theta).all():
        raise ValueError(f"Non-finite SMPL-H pose in {path}")
    if str(result["name"]) != path.stem:
        raise ValueError(f"SMPL-H name/path mismatch: {result['name']} vs {path.stem}")
    return result


def attach_synth3d_target(
    clip: CacheClip,
    annotation: Annotation,
    fit: dict[str, np.ndarray],
    *,
    fit_path: Path,
    fit_sha256: str,
    annotation_path: Path,
    annotation_sha256: str,
    offsets_sha256: str,
    phase2_split: str,
) -> CacheClip:
    indices = target_frame_indices(clip, annotation)
    theta = np.asarray(fit["thetas"], dtype=np.float32)
    if int(indices.max()) >= len(theta):
        raise IndexError(
            f"SMPL-H target out of bounds for {clip.clip_id}: "
            f"max={int(indices.max())} frames={len(theta)}"
        )
    target = theta[indices, 1:52]
    if target.shape != (len(clip.frame_names), NUM_JOINTS, 3):
        raise ValueError(f"Unexpected selected target shape: {target.shape}")

    metadata = json.loads(clip.metadata_json)
    previous_provider = str(metadata.get("target_provider", "unknown"))
    previous_type = str(metadata.get("target_type", "unknown"))
    metadata.pop("target_quality", None)
    metadata.update(
        {
            "dataset": "How2Sign",
            "target_dataset": "How2Sign-Synth3D",
            "target_provider": TARGET_PROVIDER,
            "target_type": "released_3d_smplh_pose",
            "target_scope": "SMPL-H local body and both-hand rotations (51 joints)",
            "phase2_split": phase2_split,
            "sgnify_training_reads": 0,
            "previous_target_provider": previous_provider,
            "previous_target_type": previous_type,
            "target_contract": {
                "independent_from_initializer": True,
                "released_3d_fit": True,
                "camera_synchronized": True,
                "smplh_to_refiner_mapping": "thetas[:, 1:52] -> 21 body + 15 left hand + 15 right hand",
                "frame_binding": "floor(START_aligned * clip_fps + 0.5) + frame_numbers",
                "aligned_start_seconds": annotation.start,
                "aligned_end_seconds": annotation.end,
                "clip_fps": float(clip.fps),
                "first_fit_frame": int(indices[0]),
                "last_fit_frame": int(indices[-1]),
                "fit_frames": int(len(theta)),
                "sequence_name": annotation.sequence_name,
                "fit_path": str(fit_path.resolve()),
                "fit_sha256": fit_sha256,
                "annotation_path": str(annotation_path.resolve()),
                "annotation_sha256": annotation_sha256,
                "offsets_sha256": offsets_sha256,
                "sgnify_training_reads": 0,
            },
        }
    )
    valid = np.ones((len(target), NUM_JOINTS), dtype=bool)
    return replace(
        clip,
        target_axis_angle=target.astype(np.float32),
        target_rotation_valid=valid,
        target_quality=valid.astype(np.float32),
        target_joint_positions=None,
        target_joint_valid=None,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _regional_errors(
    estimate: np.ndarray | None, exact: np.ndarray
) -> dict[str, tuple[float, int]]:
    if estimate is None:
        return {name: (0.0, 0) for name in REGIONS}
    with torch.no_grad():
        error = geodesic_distance(
            axis_angle_to_matrix(torch.from_numpy(estimate.astype(np.float32))),
            axis_angle_to_matrix(torch.from_numpy(exact.astype(np.float32))),
        ).numpy()
    return {
        name: (float(error[:, indices].sum()), int(error[:, indices].size))
        for name, indices in REGIONS.items()
    }


def _add_errors(
    totals: dict[str, dict[str, list[float | int]]],
    label: str,
    values: dict[str, tuple[float, int]],
) -> None:
    for region, (total, count) in values.items():
        totals[label][region][0] += total
        totals[label][region][1] += count


def _mean_errors(
    totals: dict[str, dict[str, list[float | int]]]
) -> dict[str, dict[str, float | None]]:
    return {
        label: {
            region: (float(total) / int(count) if int(count) else None)
            for region, (total, count) in by_region.items()
        }
        for label, by_region in totals.items()
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    template_root = args.template_root.resolve()
    smplh_root = args.smplh_root.resolve()
    annotations_root = args.annotations_root.resolve()
    output_root = args.output_root.resolve()
    offsets_path = args.offsets_csv.resolve()
    if output_root.exists():
        raise FileExistsError(f"Append-only output exists: {output_root}")
    if not smplh_root.is_dir():
        raise FileNotFoundError(smplh_root)
    if not offsets_path.is_file():
        raise FileNotFoundError(offsets_path)

    annotation_paths = {
        split: annotations_root / f"how2sign_{split}_aligned.csv"
        for split in ("train", "val", "test")
    }
    annotations = {
        split: _load_annotations(path) for split, path in annotation_paths.items()
    }
    annotation_hashes = {
        split: sha256_file(path) for split, path in annotation_paths.items()
    }
    offsets_sha256 = sha256_file(offsets_path)

    output_root.mkdir(parents=True)
    (output_root / "splits").mkdir()
    for split in SPLITS:
        (output_root / "clips" / split).mkdir(parents=True)
    incomplete = output_root / "_INCOMPLETE"
    incomplete.write_text("materialization in progress\n", encoding="utf-8")

    report: dict[str, Any] = {
        "schema_version": 1,
        "provider": TARGET_PROVIDER,
        "template_root": str(template_root),
        "smplh_root": str(smplh_root),
        "annotations_root": str(annotations_root),
        "offsets_csv": str(offsets_path),
        "offsets_sha256": offsets_sha256,
        "sgnify_training_or_selection_reads": 0,
        "splits": {},
    }

    for split in SPLITS:
        manifest = template_root / "splits" / f"{split}.json"
        paths = _manifest_paths(manifest)
        excluded: Counter[str] = Counter()
        candidates_by_fit: dict[Path, list[Candidate]] = defaultdict(list)
        for source in paths:
            clip = load_cache_clip(source)
            metadata = json.loads(clip.metadata_json)
            official_split = str(metadata.get("official_split", ""))
            if official_split not in {"train", "val", "test"}:
                excluded["invalid_official_split"] += 1
                continue
            source_clip = str(metadata.get("source_clip", clip.clip_id))
            annotation = annotations[official_split].get(source_clip)
            if annotation is None:
                excluded["annotation_missing"] += 1
                continue
            if not annotation.aligned:
                excluded["annotation_unaligned"] += 1
                continue
            fit_path = smplh_root / official_split / f"{annotation.sequence_name}.npz"
            if not fit_path.is_file():
                excluded["fit_missing"] += 1
                continue
            if not np.isfinite(clip.fps) or clip.fps <= 0:
                excluded["invalid_fps"] += 1
                continue
            candidate = Candidate(
                split=split,
                official_split=official_split,
                source=source,
                annotation=annotation,
                annotation_path=annotation_paths[official_split],
                fit_path=fit_path,
            )
            candidates_by_fit[fit_path].append(candidate)

        entries_by_source: dict[Path, str] = {}
        source_groups: set[str] = set()
        used_fits: dict[str, str] = {}
        error_totals = {
            label: {region: [0.0, 0] for region in REGIONS}
            for label in ("initializer_to_exact", "previous_target_to_exact")
        }
        frames = completed = 0
        for fit_index, (fit_path, candidates) in enumerate(
            sorted(candidates_by_fit.items(), key=lambda item: str(item[0])), start=1
        ):
            fit = _load_smplh_fit(fit_path)
            fit_sha256 = sha256_file(fit_path)
            used_fits[str(fit_path.resolve())] = fit_sha256
            for candidate in candidates:
                clip = load_cache_clip(candidate.source)
                try:
                    updated = attach_synth3d_target(
                        clip,
                        candidate.annotation,
                        fit,
                        fit_path=fit_path,
                        fit_sha256=fit_sha256,
                        annotation_path=candidate.annotation_path,
                        annotation_sha256=annotation_hashes[candidate.official_split],
                        offsets_sha256=offsets_sha256,
                        phase2_split=split,
                    )
                except IndexError:
                    excluded["target_out_of_bounds"] += 1
                    continue
                exact = updated.target_axis_angle
                assert exact is not None
                _add_errors(
                    error_totals,
                    "initializer_to_exact",
                    _regional_errors(clip.init_axis_angle, exact),
                )
                _add_errors(
                    error_totals,
                    "previous_target_to_exact",
                    _regional_errors(clip.target_axis_angle, exact),
                )
                metadata = json.loads(updated.metadata_json)
                source_group = str(metadata.get("source_group", ""))
                if not source_group:
                    raise ValueError(f"Clip lacks source_group: {clip.clip_id}")
                source_groups.add(source_group)
                destination = output_root / "clips" / split / candidate.source.name
                temporary = destination.with_name(destination.stem + ".tmp.npz")
                save_cache_clip(temporary, updated)
                os.replace(temporary, destination)
                entries_by_source[candidate.source.resolve()] = (
                    f"../clips/{split}/{destination.name}"
                )
                frames += len(updated.frame_names)
                completed += 1
                if completed % 500 == 0:
                    print(
                        f"materialize split={split} clips={completed} "
                        f"frames={frames} fits={fit_index}/{len(candidates_by_fit)}",
                        flush=True,
                    )

        entries = [
            entries_by_source[path.resolve()]
            for path in paths
            if path.resolve() in entries_by_source
        ]
        if not entries:
            raise ValueError(f"No eligible exact targets for split={split}")
        output_manifest = output_root / "splits" / f"{split}.json"
        manifest_payload = {
            "clips": entries,
            "dataset": "How2Sign",
            "target_dataset": "How2Sign-Synth3D",
            "target_provider": TARGET_PROVIDER,
            "target_type": "released_3d_smplh_pose",
            "motion_domain": "sign_language_asl",
            "split": split,
            "source_groups": sorted(source_groups),
            "sgnify_excluded": True,
        }
        output_manifest.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["splits"][split] = {
            "input_clips": len(paths),
            "clips": len(entries),
            "frames": frames,
            "source_groups": len(source_groups),
            "excluded": dict(sorted(excluded.items())),
            "regional_geodesic_radians": _mean_errors(error_totals),
            "manifest": str(output_manifest.resolve()),
            "manifest_sha256": sha256_file(output_manifest),
            "used_fits": len(used_fits),
            "used_fit_sha256": used_fits,
        }

    report_path = output_root / "materialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    (output_root / "_COMPLETE").write_text(
        f"materialization_report_sha256={sha256_file(report_path)}\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--smplh-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--offsets-csv",
        type=Path,
        default=Path(
            "data/how2sign_synth3d/metadata/"
            "how2sign-greenscreen-camera-alignment.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    report = materialize(parse_args())
    summary = {
        "schema_version": report["schema_version"],
        "provider": report["provider"],
        "sgnify_training_or_selection_reads": report[
            "sgnify_training_or_selection_reads"
        ],
        "splits": {
            name: {
                key: value
                for key, value in values.items()
                if key not in {"used_fit_sha256"}
            }
            for name, values in report["splits"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
