from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from signpccx.data.manifest import prepare_manifests, read_jsonl
from signpccx.evaluation.official import run_official_evaluator
from signpccx.evaluation.audited import evaluate_audited, paired_sign_bootstrap
from signpccx.export.preflight import preflight_sign
from signpccx.geometry.topology import load_canonical_faces, validate_faces_lock
from signpccx.io import atomic_write_json, load_config, resolve_from_config
from signpccx.model.shared_beta import forward_shared_beta_sequences
from signpccx.model.canonicalizer import calibrate_external_identity, canonical_refit_external
from signpccx.optimization.identity import calibrate_shared_beta
from signpccx.optimization.camera import calibrate_shared_camera
from signpccx.optimization.post_refine import refine_hand_contact, refine_palm_hypotheses
from signpccx.optimization.staged_fitter import calibrate_signer_full, fit_signs_full
from signpccx.teachers.h4wpp_bridge import materialize_h4wpp_sequences
from signpccx.teachers.observations import load_frame_observation, validate_h4wpp_frame
from signpccx.provenance import doctor, record_provenance
from signpccx.audit import audit_full_run


def _paths(config: dict) -> dict[str, Path]:
    return {key: resolve_from_config(config, value) for key, value in config["paths"].items()}


def _faces(config: dict, paths: dict[str, Path]):
    faces = load_canonical_faces(paths["canonical_model"])
    topology = config["topology"]
    validate_faces_lock(faces, topology["faces_sha256_int64"], int(topology["face_count"]))
    return faces


def _eval_layout(paths: dict[str, Path]) -> Path:
    return paths.get("eval_layout", paths["run_root"] / "eval_layout")


def _manifest_root(paths: dict[str, Path]) -> Path:
    return paths.get("manifest_root", paths["run_root"] / "manifests")


def command_prepare(config: dict) -> dict[str, object]:
    paths = _paths(config)
    return prepare_manifests(
        paths["image_root"],
        paths["gt_root"],
        paths["signs_file"],
        paths["segments_file"],
        paths["run_root"] / "manifests",
        int(config["data"]["expected_signs"]),
        int(config["data"]["expected_frames"]),
    )


def command_calibrate_full(config: dict, device: str) -> dict[str, object]:
    paths = _paths(config)
    return calibrate_signer_full(
        _manifest_root(paths),
        paths["h4wpp_frame_cache"],
        paths["dexavatar_initializer_root"],
        paths["smplx_model_root"],
        paths.get("full_identity", paths["run_root"] / "identity" / "S1_full.npz"),
        config,
        device,
    )


def command_fit_full(
    config: dict,
    device: str,
    signs: list[str] | None,
    limit: int | None,
    frame_ids: list[int] | None,
) -> dict[str, object]:
    paths = _paths(config)
    return fit_signs_full(
        _manifest_root(paths),
        paths["h4wpp_frame_cache"],
        paths["dexavatar_initializer_root"],
        paths.get("full_identity", paths["run_root"] / "identity" / "S1_full.npz"),
        paths["smplx_model_root"],
        paths["mano_smplx_vertex_ids"],
        paths["run_root"] / "fit_sequences",
        config,
        device,
        None if not signs else set(signs),
        limit,
        None if not frame_ids else set(frame_ids),
    )


def command_validate_h4wpp_frame(
    config: dict,
    sign: str,
    frame_id: int,
    device: str,
) -> dict[str, object]:
    paths = _paths(config)
    manifest = _manifest_root(paths) / f"{sign}.jsonl"
    matches = [record for record in read_jsonl(manifest) if record.source_frame_id == frame_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest record for {sign}/{frame_id}, got {len(matches)}")
    observation = load_frame_observation(
        matches[0], paths["h4wpp_frame_cache"], paths.get("dexavatar_initializer_root")
    )
    return validate_h4wpp_frame(
        observation, paths["smplx_model_root"], paths["run_root"] / "teacher_gate", device
    )


def command_audit_full_run(
    config: dict,
    signs: list[str] | None,
    require_evaluation: bool,
    metrics_name: str,
) -> dict[str, object]:
    paths = _paths(config)
    return audit_full_run(
        paths["run_root"], _manifest_root(paths), config["method"], paths["evaluator"],
        signs=None if not signs else set(signs), require_evaluation=require_evaluation,
        metrics_name=metrics_name,
    )


def command_materialize(config: dict) -> dict[str, object]:
    paths = _paths(config)
    return materialize_h4wpp_sequences(
        paths["h4wpp_sequence_root"],
        paths["run_root"] / "manifests",
        paths["run_root"] / "eval_layout",
        _faces(config, paths),
        config["topology"]["export_transform"],
    )


def command_calibrate_identity(config: dict) -> dict[str, object]:
    paths = _paths(config)
    identity = config["identity"]
    return calibrate_shared_beta(
        paths["h4wpp_frame_cache"],
        paths["run_root"] / "manifests",
        paths["run_root"] / "identity" / "S1.npz",
        int(identity.get("calibration_frames", 20)),
        float(identity.get("huber_delta", 1.5)),
    )


def command_forward_shared_beta(config: dict, device: str) -> dict[str, object]:
    paths = _paths(config)
    return forward_shared_beta_sequences(
        paths["h4wpp_frame_cache"],
        paths["run_root"] / "manifests",
        paths["run_root"] / "identity" / "S1.npz",
        paths["smplx_model_root"],
        paths["run_root"] / "fit_sequences",
        device,
    )


def command_calibrate_external_identity(config: dict, device: str) -> dict[str, object]:
    paths = _paths(config)
    identity = config["identity"]
    return calibrate_external_identity(
        paths["external_v1_root"],
        paths["run_root"] / "manifests",
        paths["run_root"] / "identity" / "S1.npz",
        int(identity.get("calibration_frames", 50)),
        float(identity.get("huber_delta", 1.5)),
        paths.get("smplx_model_root"),
        paths.get("mano_smplx_vertex_ids"),
        int(identity.get("canonical_refine_steps", 0)),
        float(identity.get("learning_rate", 0.01)),
        float(identity.get("beta_anchor_weight", 0.001)),
        float(identity.get("whole_mesh_weight", 0.02)),
        device,
    )


def command_canonical_refit(config: dict, device: str, signs: list[str] | None) -> dict[str, object]:
    paths = _paths(config)
    refit = config["canonical_refit"]
    return canonical_refit_external(
        paths["external_v1_root"],
        paths["run_root"] / "manifests",
        paths["run_root"] / "identity" / "S1.npz",
        paths["smplx_model_root"],
        paths["mano_smplx_vertex_ids"],
        paths["run_root"] / "fit_sequences",
        device=device,
        steps=int(refit.get("steps", 100)),
        learning_rate=float(refit.get("learning_rate", 0.01)),
        chunk_size=int(refit.get("chunk_size", 32)),
        hand_weight=float(refit.get("hand_weight", 1.0)),
        whole_mesh_weight=float(refit.get("whole_mesh_weight", 0.02)),
        pose_anchor_weight=float(refit.get("pose_anchor_weight", 1e-4)),
        max_hand_residual_mm=float(refit.get("max_hand_residual_mm", 3.0)),
        signs=None if not signs else set(signs),
    )


def command_materialize_fitted(
    config: dict,
    signs: list[str] | None = None,
    output_root: Path | None = None,
) -> dict[str, object]:
    paths = _paths(config)
    full_method = "method" in config
    return materialize_h4wpp_sequences(
        paths["run_root"] / "fit_sequences",
        _manifest_root(paths),
        output_root.resolve() if output_root is not None else paths["run_root"] / "eval_layout",
        _faces(config, paths),
        "identity",
        source_label="SIGNPCCX_FULL_STAGED" if full_method else "CANONICAL_EXTERNAL_V1_REFIT",
        schema_version=(
            "signpccx.materialized-full-staged.v1"
            if full_method else "signpccx.materialized-canonical-refit.v1"
        ),
        signs=None if not signs else set(signs),
    )


def command_refine_palm(config: dict, device: str, signs: list[str] | None) -> dict[str, object]:
    paths = _paths(config)
    hypothesis = config["hypotheses"]
    return refine_palm_hypotheses(
        paths["source_fit_root"],
        paths["run_root"] / "manifests",
        paths["h4wpp_frame_cache"],
        paths["smplx_model_root"],
        paths["run_root"] / "fit_sequences",
        device=device,
        signs=None if not signs else set(signs),
        degrees=tuple(float(value) for value in hypothesis.get("wrist_twist_degrees", [-30, 0, 30])),
    )


def command_refine_contact(config: dict, device: str, signs: list[str] | None) -> dict[str, object]:
    paths = _paths(config)
    contact = config["contact"]
    return refine_hand_contact(
        paths["source_fit_root"],
        paths["run_root"] / "manifests",
        paths["h4wpp_frame_cache"],
        paths["smplx_model_root"],
        paths["mano_smplx_vertex_ids"],
        paths["run_root"] / "fit_sequences",
        device=device,
        signs=None if not signs else set(signs),
        confidence_threshold=float(contact.get("confidence_threshold", 0.70)),
        target_distance_m=float(contact.get("target_distance_m", 0.003)),
        steps=int(contact.get("steps", 40)),
        learning_rate=float(contact.get("learning_rate", 0.001)),
    )


def command_calibrate_camera(config: dict) -> dict[str, object]:
    paths = _paths(config)
    camera = config.get("camera", {})
    return calibrate_shared_camera(
        paths["run_root"] / "manifests",
        paths["h4wpp_frame_cache"],
        paths["run_root"] / "camera" / "C1.npz",
        huber_delta_px=float(camera.get("huber_delta_px", 8.0)),
    )


def command_preflight(
    config: dict,
    signs: list[str] | None = None,
    eval_root: Path | None = None,
) -> dict[str, object]:
    paths = _paths(config)
    run_root = paths["run_root"]
    faces = _faces(config, paths)
    items = []
    manifests = sorted(_manifest_root(paths).glob("*.jsonl"))
    if signs:
        requested = set(signs)
        unknown = requested - {manifest.stem for manifest in manifests}
        if unknown:
            raise ValueError(f"Unknown signs: {sorted(unknown)}")
        manifests = [manifest for manifest in manifests if manifest.stem in requested]
    for manifest in manifests:
        count = len(read_jsonl(manifest))
        items.append(preflight_sign(
            (eval_root.resolve() if eval_root is not None else _eval_layout(paths)) / manifest.stem / "smplifyx" / "meshes",
            count,
            faces,
            bool(config["preflight"].get("require_contiguous_names", True)),
        ))
    expected_signs = len(manifests) if signs else int(config["data"]["expected_signs"])
    expected_frames = sum(len(read_jsonl(manifest)) for manifest in manifests) if signs else int(config["data"]["expected_frames"])
    frames = sum(int(item["count"]) for item in items)
    if len(items) != expected_signs or frames != expected_frames:
        raise RuntimeError(f"preflight totals {len(items)}/{frames} != {expected_signs}/{expected_frames}")
    result = {"schema_version": "signpccx.preflight.v1", "status": "ok", "signs": len(items), "frames": frames, "items": items}
    atomic_write_json(run_root / "preflight.json", result)
    return result


def command_evaluate(
    config: dict,
    python: str | None,
    evaluate_root: Path | None = None,
    sign_file: Path | None = None,
    metrics_name: str = "metrics",
) -> dict[str, object]:
    paths = _paths(config)
    return run_official_evaluator(
        paths["evaluator"],
        evaluate_root.resolve() if evaluate_root is not None else _eval_layout(paths),
        paths["gt_root"],
        sign_file.resolve() if sign_file is not None else paths["signs_file"],
        paths["segments_file"],
        paths["run_root"] / metrics_name,
        config["experiment"]["name"],
        python or sys.executable,
    )


def command_evaluate_audited(
    config: dict,
    evaluate_root: Path | None = None,
    metrics_name: str = "metrics",
    signs: list[str] | None = None,
) -> dict[str, object]:
    paths = _paths(config)
    metric_root = paths["run_root"] / metrics_name
    official = metric_root / "official_result.json"
    return evaluate_audited(
        _manifest_root(paths),
        evaluate_root.resolve() if evaluate_root is not None else _eval_layout(paths),
        paths["gt_root"],
        paths["evaluator_assets"],
        metric_root / "audited",
        config["experiment"]["name"],
        official if official.is_file() else None,
        None if not signs else set(signs),
    )


def command_bootstrap(config: dict, baseline_per_sign: Path) -> dict[str, object]:
    paths = _paths(config)
    return paired_sign_bootstrap(
        paths["run_root"] / "metrics" / "audited" / "per_sign.csv",
        baseline_per_sign.resolve(),
        paths["run_root"] / "metrics" / "paired_bootstrap.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="signpccx")
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-manifests")
    subparsers.add_parser("materialize-h4wpp")
    subparsers.add_parser("calibrate-identity")
    subparsers.add_parser("calibrate-camera")
    shared_beta = subparsers.add_parser("forward-shared-beta")
    shared_beta.add_argument("--device", default="cpu")
    external_identity = subparsers.add_parser("calibrate-external-identity")
    external_identity.add_argument("--device", default="cpu")
    canonical_refit = subparsers.add_parser("canonical-refit-external")
    canonical_refit.add_argument("--device", default="cpu")
    canonical_refit.add_argument("--sign", action="append", default=None)
    materialize_fitted = subparsers.add_parser("materialize-fitted")
    materialize_fitted.add_argument("--sign", action="append", default=None)
    materialize_fitted.add_argument("--output-root", type=Path, default=None)
    palm = subparsers.add_parser("refine-palm-hypotheses")
    palm.add_argument("--device", default="cpu")
    palm.add_argument("--sign", action="append", default=None)
    contact = subparsers.add_parser("refine-contact")
    contact.add_argument("--device", default="cpu")
    contact.add_argument("--sign", action="append", default=None)
    calibrate_full = subparsers.add_parser("calibrate-full")
    calibrate_full.add_argument("--device", default="cpu")
    fit_full = subparsers.add_parser("fit-full")
    fit_full.add_argument("--device", default="cpu")
    fit_full.add_argument("--sign", action="append", default=None)
    fit_full.add_argument("--limit", type=int, default=None)
    fit_full.add_argument("--frame-id", action="append", type=int, default=None)
    validate_frame = subparsers.add_parser("validate-h4wpp-frame")
    validate_frame.add_argument("--sign", required=True)
    validate_frame.add_argument("--frame-id", required=True, type=int)
    validate_frame.add_argument("--device", default="cpu")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--sign", action="append", default=None)
    preflight.add_argument("--eval-root", type=Path, default=None)
    evaluate = subparsers.add_parser("evaluate-official")
    evaluate.add_argument("--python", default=None)
    evaluate.add_argument("--evaluate-root", type=Path, default=None)
    evaluate.add_argument("--sign-file", type=Path, default=None)
    evaluate.add_argument("--metrics-name", default="metrics")
    audited = subparsers.add_parser("evaluate-audited")
    audited.add_argument("--evaluate-root", type=Path, default=None)
    audited.add_argument("--metrics-name", default="metrics")
    audited.add_argument("--sign", action="append", default=None)
    bootstrap = subparsers.add_parser("bootstrap-paired")
    bootstrap.add_argument("--baseline-per-sign", type=Path, required=True)
    subparsers.add_parser("record-provenance")
    subparsers.add_parser("doctor")
    audit = subparsers.add_parser("audit-full-run")
    audit.add_argument("--sign", action="append", default=None)
    audit.add_argument("--require-evaluation", action="store_true")
    audit.add_argument("--metrics-name", default="metrics")
    args = parser.parse_args()
    config = load_config(args.config)
    commands = {
        "prepare-manifests": lambda: command_prepare(config),
        "materialize-h4wpp": lambda: command_materialize(config),
        "calibrate-identity": lambda: command_calibrate_identity(config),
        "calibrate-camera": lambda: command_calibrate_camera(config),
        "forward-shared-beta": lambda: command_forward_shared_beta(config, args.device),
        "calibrate-external-identity": lambda: command_calibrate_external_identity(config, args.device),
        "canonical-refit-external": lambda: command_canonical_refit(config, args.device, args.sign),
        "materialize-fitted": lambda: command_materialize_fitted(config, args.sign, args.output_root),
        "refine-palm-hypotheses": lambda: command_refine_palm(config, args.device, args.sign),
        "refine-contact": lambda: command_refine_contact(config, args.device, args.sign),
        "calibrate-full": lambda: command_calibrate_full(config, args.device),
        "fit-full": lambda: command_fit_full(config, args.device, args.sign, args.limit, args.frame_id),
        "validate-h4wpp-frame": lambda: command_validate_h4wpp_frame(
            config, args.sign, args.frame_id, args.device
        ),
        "preflight": lambda: command_preflight(config, args.sign, args.eval_root),
        "evaluate-official": lambda: command_evaluate(
            config, args.python, args.evaluate_root, args.sign_file, args.metrics_name
        ),
        "evaluate-audited": lambda: command_evaluate_audited(
            config, args.evaluate_root, args.metrics_name, args.sign
        ),
        "bootstrap-paired": lambda: command_bootstrap(config, args.baseline_per_sign),
        "record-provenance": lambda: record_provenance(config),
        "doctor": lambda: doctor(config),
        "audit-full-run": lambda: command_audit_full_run(
            config, args.sign, args.require_evaluation, args.metrics_name
        ),
    }
    print(json.dumps(commands[args.command](), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
