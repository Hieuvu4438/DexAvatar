from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.contact_encoder import ContactTokenEncoder
from dcg_sign4d.diffusion.dposer_bridge import OfficialDPoserXBridge
from dcg_sign4d.diffusion.state_codec import StateCodec
from dcg_sign4d.diffusion.trajectory_denoiser import DPoserXConditionedTrajectoryDenoiser
from dcg_sign4d.initialization.trajectory_io import load_trajectory
from dcg_sign4d.utils.hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pinned official DPoser-X bridge audit")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--edge-count", type=int, default=60)
    parser.add_argument("--overfit-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable DPoser-X audit exists: {output}")
    if args.overfit_steps < 0:
        raise ValueError("overfit steps cannot be negative")
    if args.overfit_steps and (args.learning_rate is None or not 0 < args.learning_rate <= 0.1):
        raise ValueError("positive bounded --learning-rate is required for overfit")
    device = torch.device(args.device)
    bridge = OfficialDPoserXBridge(
        source_root=args.source_root,
        runtime_root=args.runtime_root,
        registry_path=args.registry,
        expected_commit=args.expected_commit,
        device=device,
    )
    state, metadata = load_trajectory(args.trajectory)
    state = replace(
        state,
        **{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name in state.__dataclass_fields__
            if (value := getattr(state, name)) is not None
        },
    )
    normalized = bridge.normalize_trajectory(state).reshape(-1, 256)
    time = torch.full((normalized.shape[0],), args.time, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    target = bridge.denoise_target(normalized, time, generator=generator)
    torch.manual_seed(args.seed)
    encoded, _ = StateCodec().encode(state)
    graph_state = torch.zeros(
        encoded.shape[0], encoded.shape[1], args.edge_count, dtype=torch.long, device=device
    )
    probability = torch.zeros(*graph_state.shape, 4, device=device)
    probability[..., 0] = 1
    graph = ContactGraphBatch(
        event_state=graph_state,
        event_probability=probability,
        edge_valid=torch.ones(encoded.shape[0], args.edge_count, dtype=torch.bool, device=device),
        uncertain_mask=torch.zeros_like(graph_state, dtype=torch.bool),
        segment_id=torch.zeros_like(graph_state),
        segment_duration=torch.full_like(graph_state, float(encoded.shape[1]), dtype=torch.float),
    )
    hidden_dim = 64
    token_encoder = ContactTokenEncoder(args.edge_count, hidden_dim).to(device)
    denoiser = DPoserXConditionedTrajectoryDenoiser(
        bridge,
        trajectory_steps=1000,
        hidden_dim=hidden_dim,
        heads=4,
        layers=1,
    ).to(device)
    holistic = denoiser(
        encoded,
        torch.full((encoded.shape[0],), 100, dtype=torch.long, device=device),
        token_encoder(graph, "dynamic"),
        state.valid_mask,
        torch.ones_like(state.valid_mask, dtype=encoded.dtype),
        shape=state.beta,
    )
    trainable_parameters = [
        parameter
        for parameter in [*denoiser.parameters(), *token_encoder.parameters()]
        if parameter.requires_grad
    ]
    total_parameters = sum(parameter.numel() for parameter in denoiser.parameters())
    total_parameters += sum(parameter.numel() for parameter in token_encoder.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    official_frozen = all(not parameter.requires_grad for parameter in bridge.parameters())
    overfit_report = {
        "steps": args.overfit_steps,
        "learning_rate": args.learning_rate,
        "initial_loss": None,
        "final_loss": None,
        "pass": None,
        "official_backbone_frozen": official_frozen,
    }
    if args.overfit_steps:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
        target_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        fixed_target = torch.randn(
            holistic.shape,
            generator=target_generator,
            device=device,
            dtype=holistic.dtype,
        )

        def overfit_loss() -> torch.Tensor:
            prediction = denoiser(
                encoded,
                torch.full((encoded.shape[0],), 100, dtype=torch.long, device=device),
                token_encoder(graph, "dynamic"),
                state.valid_mask,
                torch.ones_like(state.valid_mask, dtype=encoded.dtype),
                shape=state.beta,
            )
            return torch.nn.functional.mse_loss(prediction, fixed_target)

        overfit_losses = [float(overfit_loss().detach())]
        for _ in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = overfit_loss()
            loss.backward()
            optimizer.step()
            overfit_losses.append(float(overfit_loss().detach()))
        overfit_report.update(
            {
                "initial_loss": overfit_losses[0],
                "final_loss": overfit_losses[-1],
                "losses": overfit_losses,
                "pass": overfit_losses[-1] < overfit_losses[0] and official_frozen,
            }
        )
    report = {
        "schema_version": "dposer_x_runtime_audit_v1",
        "scientific_status": "IMPLEMENTATION_CANDIDATE_NOT_FROZEN",
        "source_commit": bridge.source_commit,
        "registry_status": bridge.registry_status,
        "registry_sha256": file_sha256(args.registry),
        "trajectory_sha256": metadata["trajectory_sha256"],
        "frames": normalized.shape[0],
        "dimension": normalized.shape[1],
        "time": args.time,
        "seed": args.seed,
        "normalized_finite": bool(torch.isfinite(normalized).all()),
        "score_finite": bool(torch.isfinite(target.score).all()),
        "target_finite": bool(torch.isfinite(target.normalized).all()),
        "score_abs_mean": float(target.score.abs().mean()),
        "target_abs_mean": float(target.normalized.abs().mean()),
        "safe_weights_only_loader": True,
        "holistic_candidate": {
            "trajectory_shape": list(encoded.shape),
            "output_shape": list(holistic.shape),
            "finite": bool(torch.isfinite(holistic).all()),
            "official_backbone_registered_inside_single_denoiser": True,
            "contact_edges": args.edge_count,
            "hidden_dim": hidden_dim,
            "layers": 1,
            "trained": False,
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_count,
            "tiny_overfit": overfit_report,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if args.overfit_steps and overfit_report["pass"] is not True:
        raise RuntimeError("official DPoser-X-conditioned tiny overfit did not pass")


if __name__ == "__main__":
    main()
