#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signpk.data.signavatars_dataset import SignAvatarWindowDataset, tokens_from_batch
from signpk.data.training_augmentation import augment_training_tokens
from signpk.geometry.coordinates import CameraParameters, CoordinateAdapter
from signpk.geometry.palm_frame import make_palm_frame
from signpk.geometry.robustifiers import charbonnier, geman_mcclure, masked_mean
from signpk.geometry.rotations import so3_distance
from signpk.losses.centered_vertex import centered_vertex_loss
from signpk.losses.interaction import hand_penetration_loss
from signpk.losses.kinematic import forward_kinematic_loss, palm_frame_loss, relation_loss
from signpk.losses.rotation import geodesic_rotation_loss
from signpk.losses.uncertainty import heteroscedastic_nll
from signpk.models.explicit_tokens import UPPER_BODY_INDICES
from signpk.models.palm_kinematic_coupler import PalmKinematicCoupler
from signpk.models.training_decoder import decode_pkc_center
from signpk.observers.dex_priors import DexSignPriors
from signpk.optimization.smplx_layer import SMPLXLayer
from signpk.utils.config import load_yaml, project_path
from signpk.utils.config_hash import config_hash
from signpk.utils.reproducibility import runtime_metadata, set_deterministic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cosine_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _compute_loss(
    prediction,
    decoded,
    tokens,
    batch,
    weights,
    subsets,
    device,
    faces,
    dex_priors=None,
):
    target_upper = batch["target_upper_rotmat"].to(device)
    target_left = batch["target_left_hand_rotmat"].to(device)
    target_right = batch["target_right_hand_rotmat"].to(device)
    rotation = (
        geodesic_rotation_loss(prediction.upper_rotmat, target_upper)
        + geodesic_rotation_loss(prediction.left_rotmat, target_left)
        + geodesic_rotation_loss(prediction.right_rotmat, target_right)
    ) / 3
    terms = {"rotation": rotation}
    if "target_vertices" in batch:
        target_vertices = batch["target_vertices"].to(device)
        terms["centered_ubody"] = centered_vertex_loss(
            decoded.vertices, target_vertices, subsets["upper"]
        )
        terms["centered_lhand"] = centered_vertex_loss(
            decoded.vertices, target_vertices, subsets["left"]
        )
        terms["centered_rhand"] = centered_vertex_loss(
            decoded.vertices, target_vertices, subsets["right"]
        )
        terms["uncentered_ubody"] = charbonnier(
            decoded.vertices[:, subsets["upper"]] - target_vertices[:, subsets["upper"]]
        ).mean()
    if "target_joints" in batch:
        terms["fk"] = forward_kinematic_loss(
            decoded.joints[:, :22], batch["target_joints"].to(device)[:, :22]
        )
    left_palm = right_palm = None
    if "target_left_palm" in batch and "target_right_palm" in batch:
        left_palm, _, left_valid = make_palm_frame(decoded.left_hand_joints, "left")
        right_palm, _, right_valid = make_palm_frame(decoded.right_hand_joints, "right")
        terms["palm"] = (
            palm_frame_loss(left_palm, batch["target_left_palm"].to(device), left_valid)
            + palm_frame_loss(right_palm, batch["target_right_palm"].to(device), right_valid)
        ) * 0.5
    wrist_delta = decoded.left_hand_joints[:, 0] - decoded.right_hand_joints[:, 0]
    if "target_root_rel" in batch:
        predicted_relative_palm = None
        target_relative_palm = None
        if left_palm is not None and right_palm is not None:
            predicted_relative_palm = right_palm.transpose(-1, -2) @ left_palm
            target_left_palm = batch["target_left_palm"].to(device)
            target_right_palm = batch["target_right_palm"].to(device)
            target_relative_palm = target_right_palm.transpose(-1, -2) @ target_left_palm
        terms["relation"] = relation_loss(
            wrist_delta,
            batch["target_root_rel"].to(device),
            predicted_relative_palm=predicted_relative_palm,
            target_relative_palm=target_relative_palm,
            interaction_gate=prediction.interaction_gate,
            valid=tokens.left_valid[:, tokens.timestamps.shape[1] // 2]
            & tokens.right_valid[:, tokens.timestamps.shape[1] // 2],
        )
    target_all = torch.cat([target_upper, target_left, target_right], dim=1)
    predicted_all = torch.cat(
        [prediction.upper_rotmat, prediction.left_rotmat, prediction.right_rotmat], dim=1
    )
    residual = so3_distance(predicted_all, target_all)
    logvar_all = torch.cat(
        [prediction.logvar_upper, prediction.logvar_left, prediction.logvar_right], dim=1
    )
    terms["uncertainty_nll"] = heteroscedastic_nll(residual, logvar_all)
    if left_palm is not None and right_palm is not None:
        palm_error = torch.stack(
            [
                so3_distance(left_palm, batch["target_left_palm"].to(device)),
                so3_distance(right_palm, batch["target_right_palm"].to(device)),
            ],
            dim=-1,
        )
        terms["uncertainty_nll"] = 0.5 * (
            terms["uncertainty_nll"] + heteroscedastic_nll(palm_error, prediction.logvar_palm)
        )
    if {
        "target_keypoints2d",
        "target_keypoint_confidence",
        "focal_length",
        "principal_point",
    }.issubset(batch):
        target_2d = batch["target_keypoints2d"].to(device)
        confidence = batch["target_keypoint_confidence"].to(device)
        joint_count = target_2d.shape[-2]
        projected = CoordinateAdapter.project(
            decoded.joints[:, :joint_count],
            CameraParameters(
                focal_length=batch["focal_length"].to(device),
                principal_point=batch["principal_point"].to(device),
                translation=None,
            ),
        )
        distance = torch.linalg.vector_norm(projected - target_2d, dim=-1)
        terms["reprojection_2d"] = masked_mean(
            geman_mcclure(distance, 6.0) * confidence,
            confidence > 0,
        )
    center = tokens.timestamps.shape[1] // 2
    both_valid = tokens.left_valid[:, center] & tokens.right_valid[:, center]
    terms["penetration"] = hand_penetration_loss(
        decoded.vertices,
        faces,
        subsets["left"],
        subsets["right"],
        prediction.interaction_gate,
        both_valid,
    )
    if dex_priors is not None:
        base_body = batch["base_body_rotmat"].to(device).clone()
        combined = torch.cat([tokens.upper_base_rotmat[:, center, :1], base_body], dim=1)
        combined[:, UPPER_BODY_INDICES] = prediction.upper_rotmat
        terms["sign_prior"] = dex_priors(
            combined[:, 1:], prediction.left_rotmat, prediction.right_rotmat
        )
    if "target_angular_velocity" in batch:
        terms["angular_velocity"] = (
            (prediction.angular_velocity - batch["target_angular_velocity"].to(device)).abs().mean()
        )
    if "target_wrist_velocity" in batch:
        terms["velocity"] = (
            (prediction.wrist_velocity - batch["target_wrist_velocity"].to(device)).abs().mean()
        )
    if "target_phase_gate" in batch:
        terms["phase_gate"] = torch.nn.functional.binary_cross_entropy(
            prediction.phase_gate, batch["target_phase_gate"].to(device)
        )
    if "target_interaction_gate" in batch:
        terms["interaction_gate"] = torch.nn.functional.binary_cross_entropy(
            prediction.interaction_gate, batch["target_interaction_gate"].to(device)
        )
    identity6d = prediction.upper_rot6d_residual.new_tensor([1, 0, 0, 0, 1, 0])
    terms["residual"] = torch.stack(
        [
            (prediction.upper_rot6d_residual - identity6d).square().mean(),
            (prediction.left_rot6d_residual - identity6d).square().mean(),
            (prediction.right_rot6d_residual - identity6d).square().mean(),
        ]
    ).mean()
    loss = sum(float(weights.get(name, 0.0)) * value for name, value in terms.items())
    return loss, terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PKC from frozen observer caches")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train/stage_b.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="load model weights but start a new stage")
    parser.add_argument("--seed", type=int, help="override the configured training seed")
    parser.add_argument(
        "--output-dir", type=Path, help="override the configured checkpoint directory"
    )
    parser.add_argument("--device", help="training device override, for example cpu or cuda:0")
    args = parser.parse_args()
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from are mutually exclusive")
    config = load_yaml(args.config)
    train, model_config = config["train"], config["model"]
    if bool(train.get("use_sgnify_gt", False)):
        raise ValueError("SGNify GT is forbidden for PKC training/tuning")
    seed = int(args.seed if args.seed is not None else train.get("seed", 42))
    set_deterministic(seed, deterministic=True)
    requested_device = args.device or train.get("device", "cuda")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    split = str(train.get("split", "train"))
    dataset = SignAvatarWindowDataset(project_path(train["cache_index"], PROJECT_ROOT), split=split)
    sampling_weights = torch.tensor(
        [float(row["quality_weight"]) for row in dataset.rows], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        sampling_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train["batch_size"]),
        sampler=sampler,
        num_workers=int(train.get("num_workers", 4)),
        drop_last=True,
    )
    model = PalmKinematicCoupler(**model_config).to(device)
    calibration_only = train.get("stage") == "uncertainty_calibration"
    if calibration_only and args.resume is None and args.init_from is None:
        raise ValueError(
            "Stage C calibration requires --init-from/--resume from a trained Stage B checkpoint"
        )
    if calibration_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("logvar" in name)
    geometry = config.get("geometry", {})
    model_path = project_path(
        geometry.get("smplx_model", "../data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz"),
        PROJECT_ROOT,
    )
    decoder = SMPLXLayer(model_path).to(device).eval()
    import pickle

    with project_path(
        geometry.get(
            "mano_smplx_ids", "../data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl"
        ),
        PROJECT_ROOT,
    ).open("rb") as handle:
        hand_ids = pickle.load(handle, encoding="latin1")
    subsets = {
        "left": torch.as_tensor(hand_ids["left_hand"], dtype=torch.long, device=device),
        "right": torch.as_tensor(hand_ids["right_hand"], dtype=torch.long, device=device),
        "upper": torch.as_tensor(
            __import__("numpy").load(
                project_path(
                    geometry.get(
                        "upper_body_indices",
                        "../data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint/upper_body_minus_face.npy",
                    ),
                    PROJECT_ROOT,
                )
            ),
            dtype=torch.long,
            device=device,
        ),
    }
    dex_priors = None
    if float(train["losses"].get("sign_prior", 0.0)) > 0:
        prior_config = config.get("observers", {}).get("dex_priors", {})
        if not prior_config.get("enabled", False):
            raise ValueError("sign_prior has nonzero weight but DexAvatar priors are disabled")
        dex_priors = (
            DexSignPriors(
                project_path(prior_config["body_root"], PROJECT_ROOT),
                project_path(prior_config["hand_root"], PROJECT_ROOT),
            )
            .to(device)
            .eval()
        )
    uncertainty_parameters = []
    adapter_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (uncertainty_parameters if "logvar" in name else adapter_parameters).append(parameter)
    parameter_groups = []
    if adapter_parameters:
        parameter_groups.append({"params": adapter_parameters, "lr": float(train["learning_rate"])})
    if uncertainty_parameters:
        parameter_groups.append(
            {
                "params": uncertainty_parameters,
                "lr": float(train.get("uncertainty_learning_rate", train["learning_rate"])),
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(train["weight_decay"]))
    total_steps = int(train["epochs"]) * len(loader)
    warmup = int(total_steps * float(train.get("warmup_fraction", 0.05)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_lambda(step, total_steps, warmup)
    )
    start_epoch = 0
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        if not calibration_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_path(train["output_dir"], PROJECT_ROOT)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    precision = str(train.get("precision", "fp32")).lower()
    amp_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and precision == "fp16")
    for epoch in range(start_epoch, int(train["epochs"])):
        if calibration_only:
            model.eval()
        else:
            model.train()
        running = 0.0
        for batch in loader:
            tokens = tokens_from_batch(batch, device)
            augmentation = train.get("augmentation", {})
            if not calibration_only:
                tokens = augment_training_tokens(
                    tokens,
                    observation_dropout=float(augmentation.get("observation_dropout", 0.0)),
                    feature_mask_probability=float(
                        augmentation.get("feature_mask_probability", 0.0)
                    ),
                    token_noise_std=float(augmentation.get("token_noise_std", 0.0)),
                )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction = model(tokens)
                decoded = decode_pkc_center(
                    prediction,
                    tokens,
                    batch["base_body_rotmat"].to(device),
                    batch["betas"].to(device),
                    batch["translation"].to(device),
                    decoder,
                )
                loss, _ = _compute_loss(
                    prediction,
                    decoded,
                    tokens,
                    batch,
                    train["losses"],
                    subsets,
                    device,
                    decoder.faces,
                    dex_priors,
                )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train.get("grad_clip_norm", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train.get("grad_clip_norm", 1.0))
                )
                optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config_hash": config_hash(config),
            "seed": seed,
            "device": str(device),
            "precision": precision,
            "deterministic_algorithms_warn_only": True,
            "runtime": runtime_metadata(device),
            "smplx_model_sha256": decoder.model_hash,
            "train_loss": running / len(loader),
        }
        torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, output_dir / "latest.pt")
        print(f"epoch={epoch} loss={running / len(loader):.6f}")


if __name__ == "__main__":
    main()
