"""Run only the A1R fitting stage with portable metadata and fail-fast exit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from phase2_refiner.data.a1r_contract import (
    infer_fitting_contract,
    write_fitting_contract,
)
from phase2_refiner.data.cache_schema import load_cache_clip


def run(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[2]
    fitting_root = (repository / "dexavatar_fitting").resolve()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"A1R image directory is missing: {image_root}")
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"A1R expert output must exist before fitting: {output_root}"
        )
    clip = load_cache_clip(args.cache)
    expected_images = {f"{name}{args.image_suffix}" for name in clip.frame_names}
    actual_images = {
        path.name
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() == args.image_suffix.lower()
    }
    if actual_images != expected_images:
        raise ValueError(
            "A1R input image coverage mismatch: "
            f"missing={sorted(expected_images - actual_images)[:3]} "
            f"extra={sorted(actual_images - expected_images)[:3]}"
        )
    results = output_root / "smplifyx" / "results"
    if results.exists() and any(results.iterdir()):
        raise FileExistsError(f"A1R result directory is not empty: {results}")
    contract_root = args.contract_root.resolve()
    contract = infer_fitting_contract(clip, clip.clip_id)
    contract_paths = write_fitting_contract(contract_root, contract)
    fitter_image_root = contract_root / clip.clip_id
    fitter_image_root.symlink_to(image_root, target_is_directory=True)
    command = [
        sys.executable,
        "smplifyx/main.py",
        "--config",
        str(args.config),
        "--data_folder",
        str(output_root),
        "--output_folder",
        str(output_root / "smplifyx"),
        "--img_folder",
        str(fitter_image_root),
        "--model_folder",
        "../SMPLer-X/common/utils/human_model_files",
        "--part_segm_fn",
        "assets/smplx_parts_segm.pkl",
        "--visualize",
        "False",
        "--split_num",
        "1",
        "--cur_num",
        "0",
        "--smplx_init_dir",
        args.smplx_init_dir,
        "--sign_class",
        str(contract_paths["sign_class"]),
        "--sign_segment",
        str(contract_paths["sign_segment"]),
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    subprocess.run(command, cwd=fitting_root, env=environment, check=True)
    expected = {f"{name}.pkl" for name in map(str, clip.frame_names)}
    actual = {path.name for path in results.glob("*.pkl")}
    if actual != expected:
        raise RuntimeError(
            "A1R fitter returned incomplete coverage: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={sorted(expected - actual)[:3]} extra={sorted(actual - expected)[:3]}"
        )
    (contract_root / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "clip_id": clip.clip_id,
                "frames": len(expected),
                "fitting_contract": str(contract_paths["decision"]),
                "result_directory": str(results.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("cfg_files/fit_smplx_vposer_x_ensemble.yaml"),
    )
    parser.add_argument("--smplx-init-dir", default="smplerx/smplx")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image-suffix", default=".png")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
