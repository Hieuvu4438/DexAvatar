"""Build a target-free inference cache from a selectable frozen initializer.

The default ``fitted-clean`` mode preserves External V1 exactly.  The optional
``raw-smplerx-wilor`` mode materializes the frontend used to train the
sign-domain refiner: SMPLer-X H32 body plus WiLoR hands, with independent
per-side fallback to the SMPLer-X hand pose when WiLoR has no detection.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from phase2_refiner.data.build_observation_cache import _array, _pose_from_params
from phase2_refiner.data.cache_schema import (
    PHASE2R_SEMANTIC_CONTRACT,
    load_cache_clip,
    save_cache_clip,
    validate_phase2r_semantics,
)
from phase2_refiner.geometry.rotations import matrix_to_axis_angle
from phase2_refiner.provenance import sha256_file


FITTED_CLEAN = "fitted-clean"
RAW_SMPLERX_WILOR = "raw-smplerx-wilor"


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _replace_initializer(template, result_paths: list[Path], source_manifest: Path):
    if template.target_axis_angle is not None or template.target_joint_positions is not None:
        raise ValueError(f"Template contains target fields: {template.clip_id}")
    params = [_load(path) for path in result_paths]
    pose = np.stack([_pose_from_params(item) for item in params]).astype(np.float32)
    metadata = json.loads(template.metadata_json)
    metadata.update(
        {
            "dataset": "SGNify-inference-images-only",
            "target_fields_present": False,
            "initializer_expert": "frozen WiLoR DexAvatar view with coverage fallback",
            "initializer_source_manifest": str(source_manifest.resolve()),
            "initializer_source_manifest_sha256": sha256_file(source_manifest),
            "sgnify_target_reads": 0,
            "requires_reprojection_enrichment": True,
        }
    )
    metadata.pop("reprojection_residual_provider", None)
    metadata.pop("reprojection_residual_clipping_fraction", None)
    return replace(
        template,
        init_axis_angle=pose,
        betas=np.median(
            np.stack([_array(item.get("betas"), 10) for item in params]), axis=0
        ).astype(np.float32),
        global_orient=np.stack(
            [_array(item.get("global_orient"), 3) for item in params]
        ).astype(np.float32),
        transl=np.stack([_array(item.get("transl"), 3) for item in params]).astype(
            np.float32
        ),
        jaw_pose=np.stack(
            [_array(item.get("jaw_pose"), 3) for item in params]
        ).astype(np.float32),
        leye_pose=np.stack(
            [_array(item.get("leye_pose"), 3) for item in params]
        ).astype(np.float32),
        reye_pose=np.stack(
            [_array(item.get("reye_pose"), 3) for item in params]
        ).astype(np.float32),
        expression=np.stack(
            [_array(item.get("expression"), 10) for item in params]
        ).astype(np.float32),
        source_paths=np.asarray([str(path.resolve()) for path in result_paths]),
        source_sha256=np.asarray([sha256_file(path) for path in result_paths]),
        # The historical cache predates explicit alternate-expert fields.  On
        # load it defaults alternate to the selected initializer with a false
        # validity mask; preserve that exact semantic under the current schema.
        alternate_axis_angle=pose.copy(),
        alternate_rotation_valid=np.zeros(pose.shape[:2], dtype=bool),
        reprojection_residual_2d=np.zeros_like(template.reprojection_residual_2d),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _source_index(metadata: dict, *, name: str, role: str) -> int:
    matches = [
        int(item["source_id"])
        for item in metadata.get("sources", [])
        if item.get("name") == name and item.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source name={name!r}, role={role!r}; got {matches}"
        )
    return matches[0]


def _verified_source_hash(metadata: dict, path: Path) -> str:
    expected_by_path = {
        candidate.resolve(): digest
        for candidate, digest in (
            (Path(key), value) for key, value in metadata.get("source_hashes", {}).items()
        )
    }
    expected = expected_by_path.get(path.resolve())
    if expected is None:
        raise ValueError(f"Raw SMPLer-X source is absent from cache provenance: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Raw SMPLer-X source hash mismatch: {path}")
    return actual


def _replace_raw_initializer(
    template,
    observation_dir: Path,
    smplerx_root: Path,
    smplerx_subpath: Path,
):
    """Replace a targetless template with raw H32-body/WiLoR-hand rotations."""

    if template.target_axis_angle is not None or template.target_joint_positions is not None:
        raise ValueError(f"Template contains target fields: {template.clip_id}")
    tensor_path = observation_dir / "observations.safetensors"
    metadata_path = observation_dir / "metadata.json"
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing raw observation cache under {observation_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(tensor_path) != metadata.get("artifact_sha256"):
        raise ValueError(f"Raw observation cache hash mismatch: {tensor_path}")
    values = load_file(tensor_path)
    frame_ids = values["frame_ids"].cpu().numpy().astype(np.int64)
    if not np.array_equal(frame_ids, template.frame_numbers):
        raise ValueError(
            f"Raw observation/template frame mismatch for {template.clip_id}"
        )
    rotations = values["rotations"]
    valid_rot = values["valid_rot"]
    if rotations.ndim != 5 or rotations.shape[2:] != (55, 3, 3):
        raise ValueError(f"Unexpected raw rotation shape: {tuple(rotations.shape)}")
    if valid_rot.shape != rotations.shape[:3]:
        raise ValueError(f"Unexpected raw validity shape: {tuple(valid_rot.shape)}")
    body_source = _source_index(metadata, name="smplerx", role="body_initializer")
    wilor_source = _source_index(metadata, name="wilor", role="hand_hypothesis")
    if not bool(valid_rot[:, body_source, 1:22].all()):
        raise ValueError(f"Incomplete raw SMPLer-X body rotations: {template.clip_id}")

    # Signal4D's canonical 55-joint order is global, body[21], jaw/eyes[3],
    # left hand[15], right hand[15].  Phase2R consumes body+hands only.
    initial_matrix = torch.cat(
        (
            rotations[:, body_source, 1:22],
            rotations[:, body_source, 25:40],
            rotations[:, body_source, 40:55],
        ),
        dim=1,
    ).clone()
    left_valid = valid_rot[:, wilor_source, 25:40]
    right_valid = valid_rot[:, wilor_source, 40:55]
    initial_matrix[:, 21:36] = torch.where(
        left_valid[..., None, None],
        rotations[:, wilor_source, 25:40],
        initial_matrix[:, 21:36],
    )
    initial_matrix[:, 36:51] = torch.where(
        right_valid[..., None, None],
        rotations[:, wilor_source, 40:55],
        initial_matrix[:, 36:51],
    )
    initial = matrix_to_axis_angle(initial_matrix).cpu().numpy().astype(np.float32)

    result_paths = [
        smplerx_root
        / template.clip_id
        / smplerx_subpath
        / f"low_{int(frame_id):03d}.pkl"
        for frame_id in frame_ids
    ]
    missing = [path for path in result_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    params = [_load(path) for path in result_paths]
    source_sha256 = np.asarray(
        [_verified_source_hash(metadata, path) for path in result_paths]
    )
    metadata_json = json.loads(template.metadata_json)
    left_frames = left_valid.all(dim=1).cpu().numpy()
    right_frames = right_valid.all(dim=1).cpu().numpy()
    metadata_json.update(
        {
            "dataset": "SGNify-inference-images-only",
            "target_fields_present": False,
            "initializer_mode": RAW_SMPLERX_WILOR,
            "initializer_expert": (
                "raw SMPLer-X H32 body + WiLoR hands with independent "
                "per-side SMPLer-X fallback"
            ),
            "raw_observation_cache": str(observation_dir.resolve()),
            "raw_observation_sha256": sha256_file(tensor_path),
            "raw_observation_metadata_sha256": sha256_file(metadata_path),
            "smplerx_root": str(smplerx_root.resolve()),
            "smplerx_subpath": str(smplerx_subpath),
            "wilor_fused_left_frames": int(left_frames.sum()),
            "wilor_fused_right_frames": int(right_frames.sum()),
            "wilor_left_coverage": float(left_frames.mean()),
            "wilor_right_coverage": float(right_frames.mean()),
            "sgnify_target_reads": 0,
            "requires_reprojection_enrichment": True,
        }
    )
    metadata_json.pop("reprojection_residual_provider", None)
    metadata_json.pop("reprojection_residual_clipping_fraction", None)
    fallback_reason = []
    for left_ok, right_ok in zip(left_frames, right_frames, strict=True):
        reasons = []
        if not left_ok:
            reasons.append("lhand_smplerx_fallback")
        if not right_ok:
            reasons.append("rhand_smplerx_fallback")
        fallback_reason.append(";".join(reasons))
    result = replace(
        template,
        init_axis_angle=initial,
        betas=np.median(
            np.stack([_array(item.get("betas"), 10) for item in params]), axis=0
        ).astype(np.float32),
        global_orient=np.stack(
            [_array(item.get("global_orient"), 3) for item in params]
        ).astype(np.float32),
        transl=np.stack([_array(item.get("transl"), 3) for item in params]).astype(
            np.float32
        ),
        jaw_pose=np.stack(
            [_array(item.get("jaw_pose"), 3) for item in params]
        ).astype(np.float32),
        leye_pose=np.stack(
            [_array(item.get("leye_pose"), 3) for item in params]
        ).astype(np.float32),
        reye_pose=np.stack(
            [_array(item.get("reye_pose"), 3) for item in params]
        ).astype(np.float32),
        expression=np.stack(
            [_array(item.get("expression"), 10) for item in params]
        ).astype(np.float32),
        source_paths=np.asarray([str(path.resolve()) for path in result_paths]),
        source_sha256=source_sha256,
        alternate_axis_angle=initial.copy(),
        alternate_rotation_valid=np.zeros(initial.shape[:2], dtype=bool),
        initializer_component=np.asarray(
            ["body=smplerx;hands=wilor"] * len(initial), dtype=str
        ),
        fallback_reason=np.asarray(fallback_reason, dtype=str),
        reprojection_residual_2d=np.zeros_like(template.reprojection_residual_2d),
        semantic_contract_version=PHASE2R_SEMANTIC_CONTRACT,
        metadata_json=json.dumps(metadata_json, sort_keys=True),
    )
    validate_phase2r_semantics(result)
    return result


def build(args: argparse.Namespace) -> dict:
    template_root = args.template_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    initializer_root = None
    source_manifest = None
    raw_observation_root = None
    raw_smplerx_root = None
    if args.initializer_mode == FITTED_CLEAN:
        if args.initializer_root is None or args.source_manifest is None:
            raise ValueError(
                "fitted-clean mode requires --initializer-root and --source-manifest"
            )
        initializer_root = args.initializer_root.resolve()
        source_manifest = args.source_manifest.resolve()
        with source_manifest.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        if Path(source_payload["output"]).resolve() != initializer_root:
            raise ValueError("Initializer root does not match its locked view manifest")
        if "ground_truth" in json.dumps(source_payload).lower():
            raise ValueError("Initializer manifest unexpectedly references ground truth")
    else:
        if args.raw_observation_root is None or args.raw_smplerx_root is None:
            raise ValueError(
                "raw-smplerx-wilor mode requires --raw-observation-root and "
                "--raw-smplerx-root"
            )
        raw_observation_root = args.raw_observation_root.resolve()
        raw_smplerx_root = args.raw_smplerx_root.resolve()
    template_paths = sorted((template_root / "clips").glob("*.npz"))
    if not template_paths:
        raise ValueError(f"No template clips under {template_root}")
    clip_dir = output / "clips"
    clip_dir.mkdir(parents=True)
    entries = []
    try:
        for index, template_path in enumerate(template_paths, start=1):
            template = load_cache_clip(template_path)
            if args.initializer_mode == FITTED_CLEAN:
                result_dir = (
                    initializer_root / template.clip_id / "smplifyx" / "results"
                )
                results = [result_dir / f"{name}.pkl" for name in template.frame_names]
                missing = [path for path in results if not path.is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
                clip = _replace_initializer(template, results, source_manifest)
            else:
                clip = _replace_raw_initializer(
                    template,
                    raw_observation_root / template.clip_id,
                    raw_smplerx_root,
                    args.raw_smplerx_subpath,
                )
            destination = clip_dir / template_path.name
            save_cache_clip(destination, clip)
            entries.append(
                {
                    "cache": str(Path("clips") / destination.name),
                    "clip_id": template.clip_id,
                    "frames": len(template.frame_names),
                    "has_target": False,
                    "sha256": sha256_file(destination),
                }
            )
            print(f"[initializer] {index}/{len(template_paths)} {template.clip_id}")
        report = {
            "schema_version": 1,
            "initializer_mode": args.initializer_mode,
            "template_root": str(template_root),
            "initializer_root": str(initializer_root) if initializer_root else None,
            "initializer_manifest": str(source_manifest) if source_manifest else None,
            "initializer_manifest_sha256": (
                sha256_file(source_manifest) if source_manifest else None
            ),
            "raw_observation_root": (
                str(raw_observation_root) if raw_observation_root else None
            ),
            "raw_smplerx_root": str(raw_smplerx_root) if raw_smplerx_root else None,
            "raw_smplerx_subpath": (
                str(args.raw_smplerx_subpath)
                if args.initializer_mode == RAW_SMPLERX_WILOR
                else None
            ),
            "target_reads": 0,
            "clips": entries,
            "frames": sum(item["frames"] for item in entries),
        }
        with (output / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return report
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument(
        "--initializer-mode",
        choices=(FITTED_CLEAN, RAW_SMPLERX_WILOR),
        default=FITTED_CLEAN,
    )
    parser.add_argument("--initializer-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--raw-observation-root", type=Path)
    parser.add_argument("--raw-smplerx-root", type=Path)
    parser.add_argument(
        "--raw-smplerx-subpath", type=Path, default=Path("smplerx/smplx")
    )
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
