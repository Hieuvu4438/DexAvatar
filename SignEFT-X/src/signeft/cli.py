from __future__ import annotations

import argparse
import json
from pathlib import Path

from signeft.data.manifest import prepare_manifest
from signeft.data.protocol import lock_protocol
from signeft.evaluation import evaluate_official
from signeft.export import materialize, preflight
from signeft.io_utils import load_config, resolve_path
from signeft.optim.c0 import run_c0_exact_baseline
from signeft.optim.core import run_core_refinement


def _paths(config: dict) -> dict[str, Path]:
    return {key: resolve_path(config, value) for key, value in config["paths"].items()}


def command_protocol_lock(config: dict) -> dict[str, object]:
    paths = _paths(config)
    protocol = config["protocol"]
    return lock_protocol(
        paths["signs_file"], paths["segments_file"], paths["evaluator"],
        paths["a3f_manifest_root"], paths["protocol_lock"],
        expected_evaluator_sha256=protocol["expected_evaluator_sha256"],
        expected_signs=int(protocol["expected_signs"]),
        expected_frames=int(protocol["expected_frames"]),
        reported_paper_frames=int(protocol["reported_paper_frames"]),
    )


def command_prepare_manifest(config: dict) -> dict[str, object]:
    paths = _paths(config)
    return prepare_manifest(
        paths["protocol_lock"], paths["rgb_root"], paths["a3f_run_root"],
        paths["baseline_state_root"], paths["canonical_model"], paths["manifest"],
        expected_faces_sha256=config["topology"]["faces_sha256_int64"],
    )


def command_refine(config: dict) -> dict[str, object]:
    if config["method"].get("use_gt_in_fit") is not False:
        raise RuntimeError("fitting must explicitly disable ground truth")
    if config["method"].get("temporal_pose_loss") is not False:
        raise RuntimeError("production fitting must be frame-independent")
    forbidden = [key for key in config["paths"] if "gt" in key.lower() or "evaluator_assets" in key]
    if forbidden:
        raise RuntimeError(f"fitting config contains forbidden paths: {forbidden}")
    paths = _paths(config)
    modules = config["modules"]
    unsupported = [
        name for name in ("segmentation", "pointmap", "hand_refinement")
        if modules.get(name)
    ]
    if unsupported:
        raise RuntimeError(f"modules are not promoted/wired yet: {unsupported}")
    if modules.get("heatmap"):
        return run_core_refinement(
            paths["manifest"], paths["run_root"], paths["smplx_model_root"],
            paths["pose_observation_root"],
            paths["nlf_observation_root"] if modules.get("nlf") else None,
            wrist_protection=bool(modules.get("wrist_protection")),
            device=config["runtime"]["device"],
            batch_size=int(config["runtime"]["batch_size"]),
            u1_steps=int(config["ubody"]["torso"]["steps"]),
            u2_steps=int(config["ubody"]["arms"]["steps"]),
            wrist_projection_steps=int(config["ubody"]["numerical_hand_projection_steps"]),
        )
    return run_c0_exact_baseline(paths["manifest"], paths["run_root"])


def command_materialize(config: dict) -> dict[str, object]:
    paths = _paths(config)
    return materialize(paths["manifest"], paths["run_root"], paths["run_root"] / "official_meshes")


def command_preflight(config: dict) -> dict[str, object]:
    paths = _paths(config)
    return preflight(
        paths["manifest"], paths["run_root"] / "official_meshes",
        config["topology"]["faces_sha256_int64"], paths["run_root"] / "preflight.json",
    )


def command_evaluate(config: dict, gt_root: Path, python: str | None) -> dict[str, object]:
    paths = _paths(config)
    return evaluate_official(
        paths["evaluator"], config["protocol"]["expected_evaluator_sha256"],
        paths["run_root"] / "official_meshes", gt_root.resolve(), paths["signs_file"],
        paths["segments_file"], paths["run_root"] / "metrics", config["experiment"]["name"],
        python=python or "python",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="signeft")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("protocol-lock")
    commands.add_parser("prepare-manifest")
    commands.add_parser("refine")
    commands.add_parser("materialize")
    commands.add_parser("preflight")
    evaluate = commands.add_parser("evaluate-official")
    evaluate.add_argument("--gt-root", type=Path, required=True)
    evaluate.add_argument("--python", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    handlers = {
        "protocol-lock": lambda: command_protocol_lock(config),
        "prepare-manifest": lambda: command_prepare_manifest(config),
        "refine": lambda: command_refine(config),
        "materialize": lambda: command_materialize(config),
        "preflight": lambda: command_preflight(config),
        "evaluate-official": lambda: command_evaluate(config, args.gt_root, args.python),
    }
    print(json.dumps(handlers[args.command](), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
