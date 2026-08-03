"""RDP-Fast/RDP-Best inference with exact coverage and initializer fallback."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import features_from_clip
from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from phase3_posterior.config import load_config
from phase3_posterior.data.cache_schema import load_index, load_relation_sidecar
from phase3_posterior.geometry.relation_anchors import (
    EDGE_FEATURE_DIM,
    default_edge_index,
)
from phase3_posterior.geometry.state_adapter import matrices_to_state, state_to_matrices
from phase3_posterior.losses.diffusion import SubVPSDE
from phase3_posterior.models.evidence_selector import EvidenceSelector
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.provenance import atomic_json, require_new_output, sha256_file
from phase3_posterior.sample import sample_candidates
from phase3_posterior.training import load_weights, seed_everything


def _pad(value: torch.Tensor, length: int) -> torch.Tensor:
    if len(value) == length:
        return value
    padding = [0, 0] * (value.ndim - 1) + [0, length - len(value)]
    return torch.nn.functional.pad(value, tuple(padding))


def _windows(length: int, size: int, stride: int) -> list[tuple[int, int]]:
    starts = list(range(0, max(length - size + 1, 1), stride))
    last = max(0, length - size)
    if not starts or starts[-1] != last:
        starts.append(last)
    return [(start, min(start + size, length)) for start in starts]


def _evidence(candidates: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    delta = (candidates - initial[:, None]).abs()
    body = delta[..., :21, :].mean(dim=(-3, -2, -1))
    left = delta[..., 21:36, :].mean(dim=(-3, -2, -1))
    right = delta[..., 36:51, :].mean(dim=(-3, -2, -1))
    motion = (
        (candidates[..., 1:, :, :] - candidates[..., :-1, :, :])
        .abs()
        .mean(dim=(-3, -2, -1))
    )
    finite = torch.isfinite(candidates).all(dim=(-3, -2, -1)).float()
    base = torch.stack((body, left, right, motion, finite), dim=-1)
    return torch.nn.functional.pad(base, (0, 11))


def _export_clip(output: Path, clip, state: torch.Tensor) -> list[str]:
    matrices = state_to_matrices(state.float())
    axis_angle = matrix_to_axis_angle(matrices).cpu().numpy()
    result_dir = output / clip.clip_id / "smplifyx" / "results"
    result_dir.mkdir(parents=True)
    hashes = []
    for index, frame_name in enumerate(clip.frame_names.astype(str)):
        source = Path(str(clip.source_paths[index]))
        if source.suffix != ".pkl" or not source.is_file():
            raise FileNotFoundError(
                f"Phase 3 export requires a source result PKL: {source}"
            )
        with source.open("rb") as handle:
            params = pickle.load(handle, encoding="latin1")
        params = dict(params)
        params["body_pose"] = axis_angle[index, :21].reshape(1, 63).astype(np.float32)
        params["left_hand_pose"] = (
            axis_angle[index, 21:36].reshape(1, 45).astype(np.float32)
        )
        params["right_hand_pose"] = (
            axis_angle[index, 36:51].reshape(1, 45).astype(np.float32)
        )
        target = result_dir / f"{frame_name}.pkl"
        with target.open("wb") as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        hashes.append(sha256_file(target))
    return hashes


@torch.no_grad()
def infer_clip(
    model, clip, relation, config: dict, device: torch.device
) -> torch.Tensor:
    features, initial_matrix = features_from_clip(
        clip, input_dim=int(config["model"].get("observation_dim", 45))
    )
    initial = matrices_to_state(initial_matrix).to(device)
    features = features.to(device)
    length = len(features)
    maximum = int(config["model"]["max_frames"])
    stride = maximum // 2
    candidates = int(config.get("sampling", {}).get("candidates", 4))
    steps = int(config.get("sampling", {}).get("steps", 30))
    seed = int(config.get("seed", 42))
    sde = SubVPSDE(
        **{key: config["diffusion"][key] for key in ("beta_min", "beta_max", "eps")}
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    full_noise = (
        torch.randn(
            (1, candidates - 1, length, 51, 6), generator=generator, device=device
        )
        if candidates > 1
        else None
    )
    reference_quaternion = matrix_to_quaternion(initial_matrix.to(device))
    accumulated = torch.zeros(candidates, length, 51, 4, device=device)
    weights = torch.zeros(length, device=device)
    edges = default_edge_index(device)
    for start, end in _windows(length, maximum, stride):
        window_length = end - start
        frame_valid = torch.zeros(1, maximum, dtype=torch.bool, device=device)
        frame_valid[:, :window_length] = True
        edge_features = torch.zeros(
            1, maximum, edges.shape[1], EDGE_FEATURE_DIM, device=device
        )
        edge_valid = torch.zeros(
            1, maximum, edges.shape[1], dtype=torch.bool, device=device
        )
        if relation is not None:
            edge_features[:, :window_length] = torch.from_numpy(
                relation.edge_features[start:end]
            ).to(device)
            edge_valid[:, :window_length] = torch.from_numpy(
                relation.edge_valid[start:end]
            ).to(device)
        batch = {
            "initial_state": _pad(initial[start:end], maximum)[None],
            "features": _pad(features[start:end], maximum)[None],
            "frame_valid": frame_valid,
            "edge_features": edge_features,
            "edge_index": edges[None],
            "edge_valid": edge_valid,
        }
        noise = None
        if full_noise is not None:
            noise = _pad(
                full_noise[0, :, start:end].transpose(0, 1), maximum
            ).transpose(0, 1)[None]
        sampled = sample_candidates(
            model, batch, sde, candidates, steps, seed, candidate_noise=noise
        )[0, :, :window_length]
        sampled_quaternion = matrix_to_quaternion(state_to_matrices(sampled))
        reference = reference_quaternion[start:end][None]
        hemisphere = (sampled_quaternion * reference).sum(dim=-1, keepdim=True) < 0
        sampled_quaternion = torch.where(
            hemisphere, -sampled_quaternion, sampled_quaternion
        )
        blend = torch.hann_window(window_length + 2, device=device)[1:-1].clamp_min(
            1e-3
        )
        accumulated[:, start:end] += sampled_quaternion * blend[None, :, None, None]
        weights[start:end] += blend
    quaternion = accumulated / weights[None, :, None, None].clamp_min(1e-6)
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    result = matrices_to_state(quaternion_to_matrix(quaternion))
    result[0] = initial
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True, help="Phase 3 index JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selector")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output = require_new_output(args.output)
    output.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RelationalDiffusionPosterior(config["model"]).to(device).eval()
    load_weights(model, args.checkpoint)
    selector = None
    if args.selector:
        selector_config = config.get("selector", {})
        selector = (
            EvidenceSelector(
                int(selector_config.get("feature_dim", 16)),
                int(selector_config.get("width", 128)),
            )
            .to(device)
            .eval()
        )
        load_weights(selector, args.selector, strict=True)
    reports = []
    for entry in load_index(args.cache):
        clip = load_cache_clip(entry.clip_path)
        relation = (
            load_relation_sidecar(entry.relation_path) if entry.relation_path else None
        )
        candidates = infer_clip(model, clip, relation, config, device)
        selected = 0
        if selector is not None:
            initial_matrix = axis_angle_to_matrix(
                torch.from_numpy(clip.init_axis_angle).float()
            )
            initial_state = matrices_to_state(initial_matrix).to(device)[None]
            selected = int(
                selector.select(_evidence(candidates[None], initial_state)).item()
            )
        selected_state = candidates[selected]
        fallback = not torch.isfinite(selected_state).all()
        if fallback:
            selected = 0
            selected_state = candidates[0]
        hashes = _export_clip(output, clip, selected_state.cpu())
        reports.append(
            {
                "clip_id": clip.clip_id,
                "frames": len(hashes),
                "candidate": selected,
                "fallback": fallback,
                "result_sha256": hashes,
            }
        )
    atomic_json(
        output / "phase3_diagnostics.json",
        {"clips": reports, "checkpoint_sha256": sha256_file(args.checkpoint)},
    )
    print(
        json.dumps(
            {"clips": len(reports), "frames": sum(item["frames"] for item in reports)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
