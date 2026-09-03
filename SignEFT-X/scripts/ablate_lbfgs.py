#!/usr/bin/env python3
"""Isolated L-BFGS ablation for the two SignEFT-X optimization stages.

This script intentionally lives outside ``src/signeft`` and never writes below
the released method output.  It reuses the frozen inputs and loss definitions,
but substitutes PyTorch L-BFGS with a strong-Wolfe line search for Adam.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from signeft.canonical.refinement import (
    PARAMETER_SHAPES,
    _stack_parameters,
    initializer_frame_paths,
    load_initializer_parameters,
    load_mano_smplx_ids,
    load_obj_vertices,
)
from signeft.hand.model import CanonicalBatch, FrozenSMPLX
from signeft.hand.refinement import (
    SMPLX_HAND_JOINTS,
    _load_wilor,
    palm_canonical,
)
from signeft.io.obj import load_obj, write_obj
from signeft.io_utils import atomic_write_json, load_config, sha256_file
from signeft.manifest import read_hand_manifest, read_jsonl
from signeft.model.kinematics import apply_lie_residual, so3_log_map


BOUNDARY_X180 = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def _path(config: dict, key: str) -> Path:
    return Path(config["paths"][key])


def _tensor(value: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _lbfgs(
    parameters: list[torch.nn.Parameter],
    *,
    lr: float,
    max_iter: int,
    history_size: int,
) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        parameters,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-7,
        tolerance_change=1e-12,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )


def _run_lbfgs(
    optimizer: torch.optim.LBFGS,
    closure: Callable[[], torch.Tensor],
) -> tuple[float, int, float]:
    calls = 0
    best = math.inf

    def counted() -> torch.Tensor:
        nonlocal calls, best
        calls += 1
        loss = closure()
        value = float(loss.detach())
        if math.isfinite(value):
            best = min(best, value)
        return loss

    started = time.perf_counter()
    optimizer.step(counted)
    elapsed = time.perf_counter() - started
    if not math.isfinite(best):
        raise FloatingPointError("L-BFGS produced no finite objective")
    return best, calls, elapsed


def fit_identity_lbfgs(
    config: dict,
    baseline_root: Path,
    output_root: Path,
    *,
    max_iter: int,
    lr: float,
    history_size: int,
    micro_batch: int,
) -> Path:
    """Fit the shared beta with the released identity objective and L-BFGS."""
    output_npz = output_root / "identity" / "signer.npz"
    output_json = output_npz.with_suffix(".json")
    if output_npz.is_file() and output_json.is_file():
        return output_npz

    import smplx

    baseline_npz = baseline_root / "identity" / "signer.npz"
    baseline_json = baseline_npz.with_suffix(".json")
    with np.load(baseline_npz, allow_pickle=False) as archive:
        beta_initial_np = np.asarray(archive["robust_beta"], dtype=np.float32).reshape(1, 10)
    metadata = json.loads(baseline_json.read_text(encoding="utf-8"))
    manifests = baseline_root / "manifests"
    lookup = {
        (record.sign, record.source_frame_id): record
        for path in sorted(manifests.glob("*.jsonl"))
        for record in read_jsonl(path)
    }
    frames = [
        initializer_frame_paths(
            _path(config, "initializer_root"),
            lookup[(item["sign"], int(item["frame_id"]))],
        )
        for item in metadata["selected"]
    ]
    parameters_np = _stack_parameters(frames)
    references_np = np.stack([load_obj_vertices(frame.mesh_path) for frame in frames]) * BOUNDARY_X180
    count = len(frames)
    if count != int(config["identity"]["calibration_frames"]):
        raise RuntimeError(f"calibration frame mismatch: {count}")

    device = str(config["runtime"]["device"])
    model = smplx.create(
        str(_path(config, "smplx_model_root")), model_type="smplx",
        gender="neutral", num_betas=10, use_pca=False, use_face_contour=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    left_np, right_np = load_mano_smplx_ids(_path(config, "mano_smplx_ids"))
    left_ids = torch.as_tensor(left_np, dtype=torch.long, device=device)
    right_ids = torch.as_tensor(right_np, dtype=torch.long, device=device)

    beta = torch.nn.Parameter(_tensor(beta_initial_np, device))
    left_delta = torch.nn.Parameter(torch.zeros((count, 45), device=device))
    right_delta = torch.nn.Parameter(torch.zeros((count, 45), device=device))
    optimizer = _lbfgs(
        [beta, left_delta, right_delta], lr=lr,
        max_iter=max_iter, history_size=history_size,
    )
    beta_anchor_weight = float(config["identity"]["beta_anchor_weight"])
    whole_mesh_weight = float(config["identity"]["whole_mesh_weight"])
    best_loss = math.inf
    best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def closure() -> torch.Tensor:
        nonlocal best_loss, best_state
        optimizer.zero_grad(set_to_none=True)
        total_value = 0.0
        for start in range(0, count, micro_batch):
            end = min(start + micro_batch, count)
            weight = (end - start) / count
            fixed = {
                key: _tensor(parameters_np[key][start:end], device)
                for key in (
                    "global_orient", "body_pose", "jaw_pose", "leye_pose",
                    "reye_pose", "expression", "transl",
                )
            }
            vertices = model(
                left_hand_pose=_tensor(parameters_np["left_hand_pose"][start:end], device)
                + left_delta[start:end],
                right_hand_pose=_tensor(parameters_np["right_hand_pose"][start:end], device)
                + right_delta[start:end],
                betas=beta.expand(end - start, -1), return_verts=True, **fixed,
            ).vertices
            reference = _tensor(references_np[start:end], device)
            hand_losses = []
            for indices in (left_ids, right_ids):
                prediction = vertices[:, indices]
                target = reference[:, indices]
                hand_losses.append(torch.square(
                    prediction - prediction.mean(1, keepdim=True)
                    - target + target.mean(1, keepdim=True)
                ).mean())
            hand_loss = 0.5 * (hand_losses[0] + hand_losses[1])
            whole_loss = torch.square(vertices - reference).mean()
            pose_anchor = 1e-4 * (
                torch.square(left_delta[start:end]).mean()
                + torch.square(right_delta[start:end]).mean()
            )
            local = weight * (hand_loss + whole_mesh_weight * whole_loss + pose_anchor)
            if not torch.isfinite(local):
                raise FloatingPointError("non-finite identity objective")
            local.backward()
            total_value += float(local.detach())
        anchor = beta_anchor_weight * torch.square(beta - _tensor(beta_initial_np, device)).mean()
        anchor.backward()
        total_value += float(anchor.detach())
        if total_value < best_loss:
            best_loss = total_value
            best_state = (
                beta.detach().clone(), left_delta.detach().clone(), right_delta.detach().clone()
            )
        return torch.tensor(total_value, dtype=torch.float32, device=device)

    initial_beta = beta.detach().clone()
    _, closure_calls, elapsed = _run_lbfgs(optimizer, closure)
    if best_state is None:
        raise RuntimeError("identity L-BFGS did not produce a state")
    with torch.no_grad():
        beta.copy_(best_state[0])
    shared_beta = beta.detach().cpu().numpy().reshape(10).astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        beta=shared_beta,
        robust_beta=beta_initial_np.reshape(10),
        calibration_frame_ids=np.asarray(
            [frame.record.source_frame_id for frame in frames], dtype=np.int64
        ),
    )
    atomic_write_json(output_json, {
        "schema_version": "signeft.ablation-lbfgs-identity.v1",
        "optimizer": "torch.optim.LBFGS",
        "line_search": "strong_wolfe",
        "max_iter": max_iter,
        "closure_calls": closure_calls,
        "history_size": history_size,
        "learning_rate": lr,
        "micro_batch": micro_batch,
        "wall_seconds": elapsed,
        "best_objective": best_loss,
        "beta": shared_beta.tolist(),
        "beta_l2_change": float(torch.linalg.vector_norm(beta.detach() - initial_beta).cpu()),
        "baseline_identity_sha256": sha256_file(baseline_npz),
        "objective_uses_ground_truth": False,
    })
    return output_npz


def fit_canonical_lbfgs(
    config: dict,
    baseline_root: Path,
    output_root: Path,
    identity_npz: Path,
    *,
    max_iter: int,
    lr: float,
    history_size: int,
    chunk_size: int,
    limit_signs: int | None,
) -> None:
    """Run the canonical refit with L-BFGS and write evaluator-ready meshes."""
    import smplx

    device = str(config["runtime"]["device"])
    with np.load(identity_npz, allow_pickle=False) as archive:
        shared_beta_np = np.asarray(archive["beta"], dtype=np.float32).reshape(1, 10)
    model = smplx.create(
        str(_path(config, "smplx_model_root")), model_type="smplx",
        gender="neutral", num_betas=10, use_pca=False, use_face_contour=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    left_np, right_np = load_mano_smplx_ids(_path(config, "mano_smplx_ids"))
    left_ids = torch.as_tensor(left_np, dtype=torch.long, device=device)
    right_ids = torch.as_tensor(right_np, dtype=torch.long, device=device)
    body_mask = torch.zeros((1, 63), dtype=torch.float32, device=device)
    body_mask[:, 45:63] = 1.0
    settings = config["canonicalization"]
    hand_weight = float(settings["hand_weight"])
    whole_mesh_weight = float(settings["whole_mesh_weight"])
    pose_anchor_weight = float(settings["pose_anchor_weight"])
    max_hand_residual_mm = float(settings["max_hand_residual_mm"])
    manifest_paths = sorted((baseline_root / "manifests").glob("*.jsonl"))
    if limit_signs is not None:
        manifest_paths = manifest_paths[:limit_signs]
    reports = []
    total_started = time.perf_counter()

    for manifest_path in manifest_paths:
        sign = manifest_path.stem
        report_path = output_root / "canonical" / "reports" / f"{sign}.json"
        if report_path.is_file():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        records = read_jsonl(manifest_path)
        frames = [initializer_frame_paths(_path(config, "initializer_root"), record) for record in records]
        parameters_np = _stack_parameters(frames)
        sign_reports = []
        for start in range(0, len(records), chunk_size):
            end = min(start + chunk_size, len(records))
            arrays = {key: value[start:end] for key, value in parameters_np.items()}
            reference = _tensor(
                np.stack([load_obj_vertices(frame.mesh_path) for frame in frames[start:end]])
                * BOUNDARY_X180,
                device,
            )
            body_init = _tensor(arrays["body_pose"], device)
            left_init = _tensor(arrays["left_hand_pose"], device)
            right_init = _tensor(arrays["right_hand_pose"], device)
            body_delta = torch.nn.Parameter(torch.zeros_like(body_init))
            left_delta = torch.nn.Parameter(torch.zeros_like(left_init))
            right_delta = torch.nn.Parameter(torch.zeros_like(right_init))
            fixed = {
                key: _tensor(arrays[key], device)
                for key in (
                    "global_orient", "jaw_pose", "leye_pose", "reye_pose",
                    "expression", "transl",
                )
            }
            fixed["betas"] = _tensor(shared_beta_np, device).expand(end - start, -1)
            optimizer = _lbfgs(
                [body_delta, left_delta, right_delta], lr=lr,
                max_iter=max_iter, history_size=history_size,
            )
            best_loss = math.inf
            best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            def forward() -> torch.Tensor:
                return model(
                    body_pose=body_init + body_delta * body_mask,
                    left_hand_pose=left_init + left_delta,
                    right_hand_pose=right_init + right_delta,
                    return_verts=True, **fixed,
                ).vertices

            def closure() -> torch.Tensor:
                nonlocal best_loss, best_state
                optimizer.zero_grad(set_to_none=True)
                vertices = forward()
                pred_left, pred_right = vertices[:, left_ids], vertices[:, right_ids]
                ref_left, ref_right = reference[:, left_ids], reference[:, right_ids]
                hand_loss = 0.5 * (
                    torch.square(
                        pred_left - pred_left.mean(1, keepdim=True)
                        - ref_left + ref_left.mean(1, keepdim=True)
                    ).mean()
                    + torch.square(
                        pred_right - pred_right.mean(1, keepdim=True)
                        - ref_right + ref_right.mean(1, keepdim=True)
                    ).mean()
                )
                whole_loss = torch.square(vertices - reference).mean()
                anchor = (
                    torch.square(body_delta * body_mask).mean()
                    + torch.square(left_delta).mean()
                    + torch.square(right_delta).mean()
                )
                loss = hand_weight * hand_loss + whole_mesh_weight * whole_loss + pose_anchor_weight * anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite canonical loss: {sign}[{start}:{end}]")
                loss.backward()
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value
                    best_state = (
                        body_delta.detach().clone(), left_delta.detach().clone(),
                        right_delta.detach().clone(),
                    )
                return loss

            _, closure_calls, elapsed = _run_lbfgs(optimizer, closure)
            if best_state is None:
                raise RuntimeError(f"no canonical state: {sign}[{start}:{end}]")
            with torch.no_grad():
                body_delta.copy_(best_state[0])
                left_delta.copy_(best_state[1])
                right_delta.copy_(best_state[2])
                vertices = forward()
                left_frame_mm = torch.linalg.vector_norm(
                    vertices[:, left_ids] - vertices[:, left_ids].mean(1, keepdim=True)
                    - reference[:, left_ids] + reference[:, left_ids].mean(1, keepdim=True),
                    dim=-1,
                ).mean(1) * 1000
                right_frame_mm = torch.linalg.vector_norm(
                    vertices[:, right_ids] - vertices[:, right_ids].mean(1, keepdim=True)
                    - reference[:, right_ids] + reference[:, right_ids].mean(1, keepdim=True),
                    dim=-1,
                ).mean(1) * 1000
                exported = (vertices * _tensor(BOUNDARY_X180, device)).cpu().numpy()
            if max(float(left_frame_mm.mean()), float(right_frame_mm.mean())) > max_hand_residual_mm:
                raise RuntimeError(
                    f"canonical residual gate failed: {sign}[{start}:{end}] "
                    f"{float(left_frame_mm.mean()):.3f}/{float(right_frame_mm.mean()):.3f} mm"
                )
            for local, record in enumerate(records[start:end]):
                destination = output_root / "canonical" / "meshes" / sign / f"{record.source_frame_id:06d}.obj"
                write_obj(destination, exported[local].astype(np.float32), np.asarray(model.faces, dtype=np.int64))
            sign_reports.append({
                "start": start,
                "end": end,
                "best_objective": best_loss,
                "closure_calls": closure_calls,
                "wall_seconds": elapsed,
                "left_hand_residual_mm": float(left_frame_mm.mean()),
                "right_hand_residual_mm": float(right_frame_mm.mean()),
            })
        report = {
            "sign": sign,
            "frames": len(records),
            "optimizer": "LBFGS-strong-wolfe",
            "chunks": sign_reports,
        }
        atomic_write_json(report_path, report)
        reports.append(report)
        print(f"canonical L-BFGS: {len(reports)}/{len(manifest_paths)} {sign}", flush=True)
    atomic_write_json(output_root / "canonical" / "summary.json", {
        "schema_version": "signeft.ablation-lbfgs-canonical.v1",
        "optimizer": "torch.optim.LBFGS",
        "line_search": "strong_wolfe",
        "max_iter": max_iter,
        "history_size": history_size,
        "learning_rate": lr,
        "chunk_size": chunk_size,
        "signs": len(reports),
        "frames": sum(item["frames"] for item in reports),
        "wall_seconds": time.perf_counter() - total_started,
        "identity_sha256": sha256_file(identity_npz),
        "objective_uses_ground_truth": False,
        "items": reports,
    })


def _fit_side_lbfgs(
    model: FrozenSMPLX,
    expert_joints: torch.Tensor,
    available: torch.Tensor,
    side: str,
    *,
    radius_deg: float,
    residual_prior: float,
    max_iter: int,
    lr: float,
    history_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    base_left, base_right = model.left_rotation, model.right_rotation
    base = base_left if side == "left" else base_right
    if not bool(available.any()):
        return base, torch.zeros(base.shape[:2], device=base.device), {
            "closure_calls": 0, "wall_seconds": 0.0, "best_objective": None,
        }
    delta = torch.nn.Parameter(torch.zeros(
        base.shape[0], 15, 3, dtype=base.dtype, device=base.device
    ))
    joint_ids = torch.as_tensor(SMPLX_HAND_JOINTS[side], device=base.device)
    with torch.no_grad():
        baseline = model.decode(base_left, base_right)
        reference = baseline["joints"].index_select(1, joint_ids)
        target, target_det = palm_canonical(torch.where(
            available[:, None, None], expert_joints, reference
        ))
        _, reference_det = palm_canonical(reference)
        if not bool(((target_det > 0.999) & (reference_det > 0.999)).all()):
            raise RuntimeError("degenerate palm frame")
    optimizer = _lbfgs([delta], lr=lr, max_iter=max_iter, history_size=history_size)
    radius_rad = np.deg2rad(radius_deg)
    best_energy = torch.full((base.shape[0],), float("inf"), device=base.device)
    best_rotation = base.detach().clone()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        rotation, _ = apply_lie_residual(base, delta, radius_rad)
        decoded = model.decode(
            rotation if side == "left" else base_left,
            rotation if side == "right" else base_right,
        )
        predicted, determinant = palm_canonical(decoded["joints"].index_select(1, joint_ids))
        if not bool((determinant > 0.999).all()):
            raise RuntimeError("improper predicted palm frame")
        data = F.smooth_l1_loss(
            predicted[..., 1:, :], target[..., 1:, :], reduction="none"
        ).sum(-1).mean(-1)
        prior = delta.square().sum(-1).mean(-1)
        energy = data + residual_prior * prior
        with torch.no_grad():
            improved = available & torch.isfinite(energy) & (energy < best_energy)
            best_energy[improved] = energy.detach()[improved]
            best_rotation[improved] = rotation.detach()[improved]
        weight = available.to(energy.dtype)
        loss = (energy * weight).sum() / weight.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite palm L-BFGS objective")
        loss.backward()
        return loss

    best, calls, elapsed = _run_lbfgs(optimizer, closure)
    relative = best_rotation @ base.transpose(-1, -2)
    projected, bounded = apply_lie_residual(base, so3_log_map(relative), radius_rad)
    trust = torch.rad2deg(torch.linalg.vector_norm(bounded, dim=-1))
    return projected, trust, {
        "closure_calls": calls,
        "wall_seconds": elapsed,
        "best_objective": best,
    }


def fit_palm_lbfgs(
    config: dict,
    baseline_root: Path,
    output_root: Path,
    *,
    max_iter: int,
    lr: float,
    history_size: int,
    batch_size: int,
    limit_frames: int | None,
) -> None:
    records = read_hand_manifest(baseline_root / "hand_manifest.jsonl")
    if limit_frames is not None:
        records = records[:limit_frames]
    device = str(config["runtime"]["device"])
    settings = config["hand_refinement"]
    reports = []
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        destinations = [
            output_root / "palm" / "meshes" / item.sign / f"{item.source_frame_id:06d}.obj"
            for item in batch
        ]
        if all(path.is_file() for path in destinations):
            continue
        state = CanonicalBatch.from_npz(
            [Path(item.canonical_state_path) for item in batch], device
        )
        model = FrozenSMPLX(_path(config, "smplx_model_root"), state)
        available, expert = _load_wilor(batch, _path(config, "wilor_root"), device)
        left, left_trust, left_report = _fit_side_lbfgs(
            model, expert[:, 0], available[:, 0], "left",
            radius_deg=float(settings["radius_deg"]),
            residual_prior=float(settings["residual_prior"]),
            max_iter=max_iter, lr=lr, history_size=history_size,
        )
        right, right_trust, right_report = _fit_side_lbfgs(
            model, expert[:, 1], available[:, 1], "right",
            radius_deg=float(settings["radius_deg"]),
            residual_prior=float(settings["residual_prior"]),
            max_iter=max_iter, lr=lr, history_size=history_size,
        )
        final_left = torch.where(available[:, 0, None, None, None], left, model.left_rotation)
        final_right = torch.where(available[:, 1, None, None, None], right, model.right_rotation)
        with torch.no_grad():
            final = model.decode(final_left, final_right)
        for index, (record, destination) in enumerate(zip(batch, destinations, strict=True)):
            if destination.is_file():
                continue
            if bool(available[index].any()):
                _, faces = load_obj(Path(record.canonical_obj_path))
                write_obj(
                    destination,
                    final["vertices"][index].cpu().numpy().astype(np.float32),
                    faces,
                )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(record.canonical_obj_path, destination)
        reports.append({
            "start": start,
            "end": start + len(batch),
            "left": left_report,
            "right": right_report,
            "max_left_trust_deg": float(left_trust.max()),
            "max_right_trust_deg": float(right_trust.max()),
        })
        print(f"palm L-BFGS: {start + len(batch)}/{len(records)}", flush=True)
    mesh_count = len(list((output_root / "palm" / "meshes").glob("*/*.obj")))
    atomic_write_json(output_root / "palm" / "summary.json", {
        "schema_version": "signeft.ablation-lbfgs-palm.v1",
        "optimizer": "torch.optim.LBFGS",
        "line_search": "strong_wolfe",
        "max_iter": max_iter,
        "history_size": history_size,
        "learning_rate": lr,
        "batch_size": batch_size,
        "frames": mesh_count,
        "wall_seconds": time.perf_counter() - started,
        "canonical_manifest_sha256": sha256_file(baseline_root / "hand_manifest.jsonl"),
        "objective_uses_ground_truth": False,
        "new_batches": reports,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("canonical", "palm"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/full1493"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ablation_lbfgs"))
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--identity-micro-batch", type=int, default=16)
    parser.add_argument("--limit-signs", type=int)
    parser.add_argument("--limit-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()
    torch.manual_seed(int(config.get("seed", 20260903)))
    np.random.seed(int(config.get("seed", 20260903)))
    if args.stage == "canonical":
        identity = fit_identity_lbfgs(
            config, baseline_root, output_root,
            max_iter=args.max_iter, lr=args.lr,
            history_size=args.history_size,
            micro_batch=args.identity_micro_batch,
        )
        fit_canonical_lbfgs(
            config, baseline_root, output_root, identity,
            max_iter=args.max_iter, lr=args.lr,
            history_size=args.history_size,
            chunk_size=args.batch_size,
            limit_signs=args.limit_signs,
        )
    else:
        fit_palm_lbfgs(
            config, baseline_root, output_root,
            max_iter=args.max_iter, lr=args.lr,
            history_size=args.history_size,
            batch_size=args.batch_size,
            limit_frames=args.limit_frames,
        )


if __name__ == "__main__":
    main()
