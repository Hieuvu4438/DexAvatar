"""Materialize exact frozen-A1 outputs against independent Phase-2 targets.

This command does not run external experts. It consumes their completed,
per-frame DexAvatar-compatible result PKLs and refuses to set the formal A1 bit
unless provenance is tied to the immutable G1 selection artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.build_observation_cache import _array, _pose_from_params
from phase2_refiner.data.cache_schema import load_cache_clip, save_cache_clip
from phase2_refiner.provenance import sha256_file


STACK_ID = "lane_l_a1_method_ensemble_hamer_fallback_v1"
LOCKED_VIEW_SHA256 = "cd9d52da521da5ea4cc50b3c249ff44c2f26e93380836691e9e286af96c4cb1c"
G1_EVALUATION_SHA256 = "74d3042dc872a9cf5bb87d5c6f1dff25950537d99aecadde14320f024f7180a6"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
A1_COMPONENT_SHA256 = {
    "dexavatar_fitting/cfg_files/fit_smplx_vposer_x_ensemble.yaml": (
        "d33e9593ebdc2479a9fac6d734fbdcf44ee553c7110f0b82a7febcb8304cefd2"
    ),
    "methods/Full_running_command_wilor_ensemble.sh": (
        "b1b2c8fa8d5afd462d5270595ad2293f1b0a2f27ddfd60308e2cc40247fe6846"
    ),
    "scripts/M3.5_wilor_extract.sh": (
        "2f69c89f75a5be359afeaaca0b0a419e09e9356bef61a44540f8cebd0c1dcb25"
    ),
    "scripts/M3_ensemble_init.sh": (
        "2990ad6bc2d408e33797a16736416559109599ff204fe7f32fab69c910232e42"
    ),
    "scripts/M4_smplifyx_pose_ensemble.sh": (
        "05ee0518a296c8211faf9848c1efb200d298bfc3c3a946fa7222e059022b13a8"
    ),
}


def validate_exact_a1_provenance(payload: dict[str, Any]) -> None:
    required = {
        "stack_id": STACK_ID,
        "locked_view_manifest_sha256": LOCKED_VIEW_SHA256,
        "g1_evaluation_sha256": G1_EVALUATION_SHA256,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Exact-A1 provenance {key} must be {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    component_hashes = payload.get("component_hashes")
    if component_hashes != A1_COMPONENT_SHA256:
        raise ValueError(
            "Exact-A1 component hashes do not match the frozen G1 stack"
        )


def verify_exact_a1_components(payload: dict[str, Any]) -> None:
    for name, expected in payload["component_hashes"].items():
        path = (REPOSITORY_ROOT / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Exact-A1 component is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Exact-A1 component hash changed for {name}: "
                f"expected={expected}, actual={actual}"
            )


def validate_exact_a1_result_binding(
    source_paths: np.ndarray,
    source_sha256: np.ndarray,
    result_hashes: Any,
) -> None:
    """Bind every cached initializer frame to its frozen result-PKL digest."""
    if not isinstance(result_hashes, dict) or len(result_hashes) != len(source_paths):
        raise ValueError("Exact-A1 result hash map has incomplete frame coverage")
    for source, digest in zip(source_paths, source_sha256, strict=True):
        expected = result_hashes.get(Path(str(source)).name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Exact-A1 result hash missing for {source}")
        if str(digest) != expected:
            raise ValueError(f"Exact-A1 cached source digest mismatch for {source}")


def _load_result(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _materialize_clip(template, results_dir: Path, provenance: dict[str, Any]):
    paths = [results_dir / f"{name}.pkl" for name in template.frame_names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Exact-A1 coverage failure for {template.clip_id}: "
            f"{len(missing)} missing; first={missing[0]}"
        )
    params = [_load_result(path) for path in paths]
    poses = np.stack([_pose_from_params(item) for item in params]).astype(np.float32)
    globals_ = np.stack([_array(item.get("global_orient"), 3) for item in params])
    translations = np.stack([_array(item.get("transl"), 3) for item in params])
    jaws = np.stack([_array(item.get("jaw_pose"), 3) for item in params])
    leyes = np.stack([_array(item.get("leye_pose"), 3) for item in params])
    reyes = np.stack([_array(item.get("reye_pose"), 3) for item in params])
    expressions = np.stack([_array(item.get("expression"), 10) for item in params])
    betas = np.median(
        np.stack([_array(item.get("betas"), 10) for item in params]), axis=0
    ).astype(np.float32)
    metadata = json.loads(template.metadata_json)
    metadata.update(
        {
            "initializer_expert": "exact frozen Lane-L A1 ensemble/fallback stack",
            "initializer_matches_locked_lane_a1": True,
            "initializer_provenance": provenance,
            "initializer_result_set_sha256": {
                path.name: sha256_file(path) for path in paths
            },
            "requires_reprojection_enrichment": True,
        }
    )
    metadata.pop("reprojection_residual_provider", None)
    metadata.pop("reprojection_residual_clipping_fraction", None)
    return replace(
        template,
        init_axis_angle=poses,
        betas=betas,
        global_orient=globals_.astype(np.float32),
        transl=translations.astype(np.float32),
        jaw_pose=jaws.astype(np.float32),
        leye_pose=leyes.astype(np.float32),
        reye_pose=reyes.astype(np.float32),
        expression=expressions.astype(np.float32),
        source_paths=np.asarray([str(path.resolve()) for path in paths]),
        source_sha256=np.asarray([sha256_file(path) for path in paths]),
        reprojection_residual_2d=np.zeros_like(template.reprojection_residual_2d),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    template_root = args.template_root.resolve()
    exact_root = args.exact_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only output exists: {output}")
    with args.provenance.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    validate_exact_a1_provenance(provenance)
    verify_exact_a1_components(provenance)
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "stack_id": STACK_ID,
        "template_root": str(template_root),
        "exact_root": str(exact_root),
        "provenance": str(args.provenance.resolve()),
        "provenance_sha256": sha256_file(args.provenance),
        "splits": {},
    }
    try:
        (output / "splits").mkdir()
        for split in ("train", "val", "calibration"):
            source_manifest = template_root / "splits" / f"{split}.json"
            paths = _manifest_paths(source_manifest)
            clip_dir = output / "clips" / split
            clip_dir.mkdir(parents=True)
            entries = []
            frames = 0
            for index, path in enumerate(paths, start=1):
                template = load_cache_clip(path)
                results_dir = exact_root / template.clip_id / "smplifyx" / "results"
                clip = _materialize_clip(template, results_dir, provenance)
                destination = clip_dir / f"{clip.clip_id}.npz"
                save_cache_clip(destination, clip)
                entries.append(str(Path("..") / "clips" / split / destination.name))
                frames += len(clip.frame_names)
                if index % 10 == 0 or index == len(paths):
                    print(
                        f"[exact-a1] split={split} clips={index}/{len(paths)} ",
                        flush=True,
                    )
            with source_manifest.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["clips"] = entries
            manifest["initializer_expert"] = (
                "exact frozen Lane-L A1 ensemble/fallback stack"
            )
            manifest["initializer_matches_locked_lane_a1"] = True
            manifest["requires_reprojection_enrichment"] = True
            with (output / "splits" / f"{split}.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            report["splits"][split] = {"clips": len(entries), "frames": frames}
        with (output / "materialization_report.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        shutil.rmtree(output)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
