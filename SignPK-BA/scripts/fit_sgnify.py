#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.data.cache_schema import BodyObservation, CouplerPrediction, HandObservation
from signpk.data.frame_manifest import SignManifest
from signpk.data.window_sampler import all_windows
from signpk.export.smplx_export import export_mesh_sequence, export_state_sequence
from signpk.export.diagnostics import write_jsonl
from signpk.models.explicit_tokens import ExplicitTokenBuilder, observer_disagreement
from signpk.models.palm_kinematic_coupler import PalmKinematicCoupler
from signpk.observers.dex_priors import DexSignPriors
from signpk.observers.h4w_wrapper import load_h4w_cache
from signpk.observers.omnihands_wrapper import load_omnihands_cache
from signpk.optimization.clip_ba import ClipBundleAdjuster
from signpk.optimization.factors import FactorInputs
from signpk.optimization.smplx_layer import SMPLXLayer
from signpk.optimization.state import initialize_state
from signpk.utils.config import load_yaml, project_path
from signpk.utils.config_hash import config_hash, sha256_file
from signpk.utils.reproducibility import runtime_metadata, set_deterministic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _map_dataclass(value, function):
    return type(value)(
        **{
            field.name: (
                None if getattr(value, field.name) is None else function(getattr(value, field.name))
            )
            for field in fields(value)
        }
    )


def _to(value, device: torch.device):
    return _map_dataclass(value, lambda tensor: tensor.to(device))


def _slice(value, indices: tuple[int, ...]):
    index = torch.tensor(
        indices, dtype=torch.long, device=getattr(value, fields(value)[0].name).device
    )
    return _map_dataclass(value, lambda tensor: tensor.index_select(0, index))


def _load_pkc(path: Path, model_config: dict, device: torch.device) -> PalmKinematicCoupler:
    if not path.is_file():
        raise FileNotFoundError(f"PKC checkpoint is required for learned modes: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model = PalmKinematicCoupler(**model_config).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"PKC checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()


def _predict_clip(
    model: PalmKinematicCoupler,
    body: BodyObservation,
    h4w_left: HandObservation,
    h4w_right: HandObservation,
    omni_left: HandObservation,
    omni_right: HandObservation,
    root_rel: torch.Tensor,
    timestamps: torch.Tensor,
    window_size: int,
    gap: int,
    padding: str,
    class_name: str,
) -> CouplerPrediction:
    builder = ExplicitTokenBuilder()
    outputs = []
    specs = all_windows(len(timestamps), window_size, gap, padding)
    handedness = torch.tensor([1.0 if class_name == "0" else 0.0], device=timestamps.device)
    with torch.inference_mode():
        for spec in specs:
            tokens = builder.build(
                _slice(body, spec.indices),
                _slice(h4w_left, spec.indices),
                _slice(h4w_right, spec.indices),
                _slice(omni_left, spec.indices),
                _slice(omni_right, spec.indices),
                root_rel[list(spec.indices)],
                timestamps[list(spec.indices)],
                handedness,
            )
            outputs.append(model(tokens))
    upper = torch.cat([item.upper_rotmat for item in outputs], dim=0)
    left = torch.cat([item.left_rotmat for item in outputs], dim=0)
    right = torch.cat([item.right_rotmat for item in outputs], dim=0)
    return CouplerPrediction(
        root_rotmat=upper[:, 0],
        upper_rotmat=upper,
        left_hand_rotmat=left,
        right_hand_rotmat=right,
        angular_velocity=torch.cat([item.angular_velocity for item in outputs], dim=0),
        wrist_velocity=torch.cat([item.wrist_velocity for item in outputs], dim=0),
        log_variance={
            "upper": torch.cat([item.logvar_upper for item in outputs], dim=0),
            "left": torch.cat([item.logvar_left for item in outputs], dim=0),
            "right": torch.cat([item.logvar_right for item in outputs], dim=0),
            "palm": torch.cat([item.logvar_palm for item in outputs], dim=0),
        },
        phase_gate=torch.cat([item.phase_gate for item in outputs], dim=0),
        interaction_gate=torch.cat([item.interaction_gate for item in outputs], dim=0),
    )


def _load_hand_indices(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    return (
        torch.as_tensor(data["left_hand"], dtype=torch.long, device=device),
        torch.as_tensor(data["right_hand"], dtype=torch.long, device=device),
    )


def _write_prediction_diagnostics(
    output_root: Path,
    manifest: SignManifest,
    prediction: CouplerPrediction,
    h4w_left: HandObservation,
    h4w_right: HandObservation,
    omni_left: HandObservation,
    omni_right: HandObservation,
) -> None:
    disagreement = torch.stack(
        [
            observer_disagreement(h4w_left, omni_left),
            observer_disagreement(h4w_right, omni_right),
        ],
        dim=1,
    )
    wrist_speed = torch.linalg.vector_norm(prediction.wrist_velocity, dim=-1).mean(-1)
    angular_speed = torch.linalg.vector_norm(prediction.angular_velocity, dim=-1).mean(-1)
    vertex_disagreement = disagreement[..., 0].mean(-1)
    wrist_threshold = wrist_speed.median()
    disagreement_threshold = vertex_disagreement.median()
    rows = []
    for index, record in enumerate(manifest.records):
        interaction_gate = float(prediction.interaction_gate[index].detach().cpu().squeeze())
        phase_gate = float(prediction.phase_gate[index].detach().cpu().squeeze())
        rows.append(
            {
                "frame_id": record.prediction_frame_id,
                "video_frame_id": record.video_frame_id,
                "timestamp_sec": record.timestamp_sec,
                "handedness": "one_hand" if manifest.handedness_class == "0" else "two_hand",
                "interaction": "close" if interaction_gate >= 0.5 else "separated",
                "velocity": "high" if wrist_speed[index] >= wrist_threshold else "low",
                "disagreement": (
                    "high" if vertex_disagreement[index] >= disagreement_threshold else "low"
                ),
                "segment": (
                    "early"
                    if index * 3 < len(manifest.records)
                    else "middle"
                    if index * 3 < 2 * len(manifest.records)
                    else "late"
                ),
                "phase_gate": phase_gate,
                "interaction_gate": interaction_gate,
                "wrist_speed_mps": float(wrist_speed[index].detach().cpu()),
                "angular_speed_radps": float(angular_speed[index].detach().cpu()),
                "observer_vertex_disagreement_m": float(vertex_disagreement[index].detach().cpu()),
                "observer_palm_disagreement_rad": float(
                    disagreement[index, :, 1].mean().detach().cpu()
                ),
                "left_valid": bool(omni_left.valid[index].detach().cpu()),
                "right_valid": bool(omni_right.valid[index].detach().cpu()),
                "left_padding_ratio": float(omni_left.padding_ratio[index].detach().cpu()),
                "right_padding_ratio": float(omni_right.padding_ratio[index].detach().cpu()),
            }
        )
    write_jsonl(output_root / manifest.sign_name / "frame_diagnostics.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SignPK-BA inference")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/fit/signpk_ba.yaml")
    parser.add_argument(
        "--mode", choices=["h4w_init", "pkc_feedforward", "signpk_ba"], required=True
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sign", action="append")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--omni-cache-root",
        type=Path,
        help="override the configured OmniHands cache root (useful for immutable cache snapshots)",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--max-stage-iterations",
        type=int,
        help="cap every BA stage for integration smoke tests; omitted for protocol runs",
    )
    args = parser.parse_args()
    fit_config = load_yaml(args.config)
    data_config = load_yaml(project_path(fit_config["data_config"], PROJECT_ROOT))
    model_config = load_yaml(project_path(fit_config["model_config"], PROJECT_ROOT))["model"]
    seed = 42
    set_deterministic(seed, deterministic=True)
    requested_device = args.device or fit_config["fit"].get("device", "cuda")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    data, observers, geometry = (
        data_config["data"],
        data_config["observers"],
        data_config["geometry"],
    )
    manifest_root = project_path(data["manifest_root"], PROJECT_ROOT)
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else project_path(fit_config["fit"]["output_root"], PROJECT_ROOT) / args.mode
    )
    omni_cache_root = (
        args.omni_cache_root.resolve()
        if args.omni_cache_root
        else project_path(observers["omnihands"]["cache_root"], PROJECT_ROOT)
    )
    paths = sorted(manifest_root.glob("*/manifest.json"))
    if args.sign:
        names = set(args.sign)
        paths = [path for path in paths if path.parent.name in names]
    if not paths:
        raise FileNotFoundError(f"no selected manifests in {manifest_root}")
    reference = torch.from_numpy(
        __import__("numpy")
        .load(project_path(geometry["smplx_model"], PROJECT_ROOT), allow_pickle=True)["f"]
        .astype("int64")
    )
    export_transform = torch.as_tensor(
        geometry.get("benchmark_export_rotation", torch.eye(3).tolist()), dtype=torch.float32
    )
    pkc_model = None
    smplx_layer = None
    dex_priors = None
    if args.mode != "h4w_init":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for PKC modes")
        pkc_model = _load_pkc(args.checkpoint.resolve(), model_config, device)
        smplx_layer = (
            SMPLXLayer(project_path(geometry["smplx_model"], PROJECT_ROOT)).to(device).eval()
        )
        left_indices, right_indices = _load_hand_indices(
            project_path(geometry["mano_smplx_ids"], PROJECT_ROOT), device
        )
        prior_config = observers.get("dex_priors", {})
        if args.mode == "signpk_ba" and prior_config.get("enabled", False):
            dex_priors = (
                DexSignPriors(
                    project_path(prior_config["body_root"], PROJECT_ROOT),
                    project_path(prior_config["hand_root"], PROJECT_ROOT),
                )
                .to(device)
                .eval()
            )
    for manifest_path in paths:
        manifest = SignManifest.load(manifest_path, validate_paths=True)
        body, h4w_left, h4w_right, h4w_meta = load_h4w_cache(
            project_path(observers["h4w"]["cache_root"], PROJECT_ROOT),
            manifest,
            expected_commit=observers["h4w"].get("expected_commit"),
        )
        metadata = {
            "method": args.mode,
            "seed": seed,
            "config_hash": config_hash(fit_config),
            "h4w": h4w_meta,
            "frame_identity": "manifest prediction_frame_id equals gt_frame_id",
            "used_sgnify_gt_for_fitting": False,
            "data_config_hash": config_hash(data_config),
            "model_config_hash": config_hash(model_config),
            "runtime": runtime_metadata(device),
        }
        if args.mode == "h4w_init":
            exported_vertices = torch.einsum("ij,tvj->tvi", export_transform, body.vertices)
            metadata["benchmark_export_rotation"] = export_transform.tolist()
            export_mesh_sequence(output_root, manifest, exported_vertices, reference, metadata)
            print(f"[done] {manifest.sign_name}: h4w_init")
            continue
        assert pkc_model is not None and smplx_layer is not None
        omni_path = omni_cache_root / manifest.sign_name / "omni.pt"
        omni_left, omni_right, root_rel, omni_meta = load_omnihands_cache(omni_path, manifest)
        body = _to(body, device)
        h4w_left, h4w_right = _to(h4w_left, device), _to(h4w_right, device)
        omni_left, omni_right = _to(omni_left, device), _to(omni_right, device)
        root_rel = root_rel.to(device)
        timestamps = torch.tensor([row.timestamp_sec for row in manifest.records], device=device)
        prediction = _predict_clip(
            pkc_model,
            body,
            h4w_left,
            h4w_right,
            omni_left,
            omni_right,
            root_rel,
            timestamps,
            int(data.get("temporal_window", 9)),
            int(data.get("temporal_gap", 1)),
            data.get("boundary_padding", "reflect"),
            manifest.handedness_class,
        )
        _write_prediction_diagnostics(
            output_root,
            manifest,
            prediction,
            h4w_left,
            h4w_right,
            omni_left,
            omni_right,
        )
        state = initialize_state(
            prediction.upper_rotmat,
            body.body_rotmat,
            prediction.left_hand_rotmat,
            prediction.right_hand_rotmat,
            body.shape,
            body.translation,
        ).to(device)
        metadata["omnihands"] = omni_meta
        metadata["pkc_checkpoint"] = str(args.checkpoint.resolve())
        metadata["pkc_checkpoint_sha256"] = sha256_file(args.checkpoint.resolve())
        metadata["smplx"] = {
            "model_sha256": smplx_layer.model_hash,
            "gender": "neutral",
            "use_pca": False,
            "flat_hand_mean": True,
            "num_betas": 10,
            "vertex_count": 10475,
            "mano_smplx_ids_sha256": sha256_file(
                project_path(geometry["mano_smplx_ids"], PROJECT_ROOT)
            ),
        }
        if dex_priors is not None:
            metadata["dex_priors"] = dex_priors.metadata
        metadata["benchmark_export_rotation"] = export_transform.tolist()
        metadata["frame_diagnostics"] = "frame_diagnostics.jsonl"
        if args.mode == "pkc_feedforward":
            with torch.inference_mode():
                output = smplx_layer(state)
            export_state_sequence(
                output_root,
                manifest,
                state,
                output,
                smplx_layer.faces,
                metadata,
                vertex_transform=export_transform,
            )
        else:
            ba = fit_config["bundle_adjustment"]
            stages = [dict(stage) for stage in ba["stages"]]
            if args.max_stage_iterations is not None:
                if args.max_stage_iterations < 1:
                    raise ValueError("--max-stage-iterations must be positive")
                for stage in stages:
                    stage["iterations"] = min(int(stage["iterations"]), args.max_stage_iterations)
                metadata["smoke_iteration_cap"] = args.max_stage_iterations
            adjuster = ClipBundleAdjuster(
                smplx_layer,
                stages,
                scales={
                    "hand_2d_px": float(fit_config["robustifiers"]["hand_2d_px"]),
                    "hand_vertex_m": float(fit_config["robustifiers"]["hand_vertex_m"]),
                    "wrist_relation_m": float(fit_config["robustifiers"]["wrist_relation_m"]),
                },
                grad_clip_norm=float(ba.get("grad_clip_norm", 1.0)),
                sanity=fit_config.get("fallback", {}),
            )
            result = adjuster.optimize(
                state,
                FactorInputs(
                    h4w_body=body,
                    h4w_left=h4w_left,
                    h4w_right=h4w_right,
                    omni_left=omni_left,
                    omni_right=omni_right,
                    pkc=prediction,
                    left_vertex_indices=left_indices,
                    right_vertex_indices=right_indices,
                    root_rel=root_rel,
                    timestamps=timestamps,
                    dex_priors=dex_priors,
                    faces=smplx_layer.faces,
                    one_hand_dominant=(
                        manifest.dominant_hand if manifest.handedness_class == "0" else None
                    ),
                ),
            )
            metadata["fallback_events"] = result.fallback_events
            export_state_sequence(
                output_root,
                manifest,
                result.state,
                result.output,
                smplx_layer.faces,
                metadata,
                vertex_transform=export_transform,
            )
            log_path = output_root / manifest.sign_name / "ba_factors.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(asdict(row), sort_keys=True) for row in result.records) + "\n",
                encoding="utf-8",
            )
        print(f"[done] {manifest.sign_name}: {args.mode}")


if __name__ == "__main__":
    main()
