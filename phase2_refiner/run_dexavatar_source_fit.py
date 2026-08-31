"""Run the frozen DexAvatar SignBPoser/SignHPoser fitter on source adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from phase2_refiner.provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
FITTING_ROOT = REPO_ROOT / "dexavatar_fitting"
MAIN = FITTING_ROOT / "smplifyx" / "main.py"
DEFAULT_CONFIG = FITTING_ROOT / "cfg_files" / "fit_smplx_vposer_x.yaml"
MODEL_ROOT = REPO_ROOT / "SMPLer-X" / "common" / "utils" / "human_model_files"
SEGMENTATION_CANDIDATES = (
    FITTING_ROOT / "assets" / "smplx_parts_segm.pkl",
    REPO_ROOT
    / "scratch/maps_sign_runtime_code/Ablehnen/dexavatar_fitting/assets/smplx_parts_segm.pkl",
    REPO_ROOT.parent
    / "DexAvatar-kaustesseract-probes/dexavatar_fitting/assets/smplx_parts_segm.pkl",
)
SCHEMA = "signal4d-dexavatar-source-fit-v1"


def _environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    paths = [str(FITTING_ROOT / "smplifyx"), str(FITTING_ROOT)]
    inherited = environment.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _command(
    input_root: Path,
    output_root: Path,
    clip_id: str,
    python: Path,
    segmentation: Path,
    config: Path,
) -> list[str]:
    clip_input = input_root / "clips" / clip_id
    return [
        str(python),
        str(MAIN),
        "--config",
        str(config),
        "--data_folder",
        str(clip_input),
        "--output_folder",
        str(output_root / clip_id),
        "--img_folder",
        str(clip_input / clip_id),
        "--model_folder",
        str(MODEL_ROOT),
        "--part_segm_fn",
        str(segmentation),
        "--visualize",
        "False",
        "--split_num",
        "1",
        "--cur_num",
        "0",
        "--smplx_init_dir",
        "smplerx/smplx",
        "--sign_class",
        str(input_root / "signs.txt"),
        "--sign_segment",
        str(input_root / "segment.json"),
    ]


def _render(command: list[str], environment: dict[str, str]) -> str:
    import shlex

    prefix = (
        f"PYTHONPATH={shlex.quote(environment['PYTHONPATH'])} "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(environment['CUDA_VISIBLE_DEVICES'])}"
    )
    return prefix + " " + shlex.join(command)


def run(args: argparse.Namespace) -> dict:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    python = args.python.resolve()
    config = args.config.resolve()
    source_report_path = input_root / "materialization_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("target_fields_read") is not False:
        raise ValueError("DexAvatar source inputs do not prove target independence")
    clips = [item["clip_id"] for item in source_report["clip_reports"]]
    if args.clip:
        requested = set(args.clip)
        missing = requested - set(clips)
        if missing:
            raise ValueError(f"Unknown source clips: {sorted(missing)}")
        clips = [clip for clip in clips if clip in requested]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    if not clips:
        raise ValueError("No clips selected")
    if not python.is_file():
        raise FileNotFoundError(python)
    segmentation = next(
        (path.resolve() for path in SEGMENTATION_CANDIDATES if path.is_file()), None
    )
    if segmentation is None:
        raise FileNotFoundError(
            "Missing required smplx_parts_segm.pkl; checked: "
            + ", ".join(str(path) for path in SEGMENTATION_CANDIDATES)
        )
    required_assets = (
        MAIN,
        config,
        MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz",
        FITTING_ROOT / "smplifyx/signbposer/snapshots/TR00_E078.pt",
        FITTING_ROOT / "smplifyx/signhposer/signhposer/snapshots/TR00_E100.pt",
        input_root / "signs.txt",
        input_root / "segment.json",
    )
    missing_assets = [str(path) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise FileNotFoundError(f"DexAvatar preflight assets missing: {missing_assets}")
    if output_root.exists():
        raise FileExistsError(f"Append-only DexAvatar fit root exists: {output_root}")

    environment = _environment(args.gpu)
    commands = [
        _command(input_root, output_root, clip, python, segmentation, config)
        for clip in clips
    ]
    rendered = [_render(command, environment) for command in commands]
    if args.dry_run:
        check = subprocess.run(
            [str(python), "-c", "import rewrite_body_model, data_parser"],
            cwd=FITTING_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            raise RuntimeError(
                "DexAvatar import preflight failed:\n" + check.stdout + check.stderr
            )
        return {
            "schema": SCHEMA,
            "dry_run": True,
            "import_preflight": "passed",
            "commands": rendered,
            "segmentation": str(segmentation),
            "segmentation_sha256": sha256_file(segmentation),
        }

    output_root.mkdir(parents=True)
    incomplete = output_root / ".fit_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    started = time.time()
    rows = []
    for index, (clip, command, shell_command) in enumerate(
        zip(clips, commands, rendered), start=1
    ):
        print(f"[dexavatar-fit] {index}/{len(clips)} {clip}", flush=True)
        print(shell_command, flush=True)
        clip_started = time.time()
        completed = subprocess.run(command, cwd=FITTING_ROOT, env=environment)
        if completed.returncode != 0:
            raise RuntimeError(
                f"DexAvatar fitter failed for {clip}: exit={completed.returncode}"
            )
        expected = next(
            item["frames"]
            for item in source_report["clip_reports"]
            if item["clip_id"] == clip
        )
        result_paths = sorted((output_root / clip / "results").glob("*.pkl"))
        if len(result_paths) != expected:
            raise RuntimeError(
                f"Incomplete DexAvatar fit for {clip}: {len(result_paths)}/{expected} results"
            )
        rows.append(
            {
                "clip_id": clip,
                "frames": expected,
                "elapsed_seconds": time.time() - clip_started,
                "result_sha256": [sha256_file(path) for path in result_paths],
                "command_sha256": hashlib.sha256(shell_command.encode()).hexdigest(),
            }
        )
    report = {
        "schema": SCHEMA,
        "dry_run": False,
        "clips": len(rows),
        "frames": sum(item["frames"] for item in rows),
        "elapsed_seconds": time.time() - started,
        "target_fields_read": False,
        "sgnify_labels_read": False,
        "source_report": str(source_report_path),
        "source_report_sha256": sha256_file(source_report_path),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "segmentation": str(segmentation),
        "segmentation_sha256": sha256_file(segmentation),
        "signbposer_sha256": sha256_file(
            FITTING_ROOT / "smplifyx/signbposer/snapshots/TR00_E078.pt"
        ),
        "signhposer_sha256": sha256_file(
            FITTING_ROOT / "smplifyx/signhposer/signhposer/snapshots/TR00_E100.pt"
        ),
        "clip_reports": rows,
    }
    report_path = output_root / "fit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/haipd/miniconda3/envs/dexavatar/bin/python"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--clip", action="append")
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))
