"""Generate R7 selector supervision from a frozen R6 posterior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from phase2_refiner.geometry.rotations import geodesic_distance
from phase3_posterior.config import load_config
from phase3_posterior.data.dataset import Phase3Dataset, collate_phase3
from phase3_posterior.geometry.state_adapter import state_to_matrices
from phase3_posterior.infer import _evidence
from phase3_posterior.losses.diffusion import SubVPSDE
from phase3_posterior.models.relational_diffusion import RelationalDiffusionPosterior
from phase3_posterior.provenance import sha256_file
from phase3_posterior.sample import sample_candidates
from phase3_posterior.training import load_weights, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Phase3Dataset(
        config["data"]["train_index"],
        int(config["model"]["max_frames"]),
        training=False,
        seed=seed,
        input_dim=int(config["model"].get("observation_dim", 45)),
        identity_target=bool(config["data"].get("identity_target", False)),
    )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator)
    subset = Subset(dataset, permutation[: min(args.samples, len(dataset))].tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_phase3,
    )
    model = RelationalDiffusionPosterior(config["model"]).to(device).eval()
    load_weights(model, args.checkpoint)
    sde = SubVPSDE(
        **{key: config["diffusion"][key] for key in ("beta_min", "beta_max", "eps")}
    )
    feature_rows, error_rows = [], []
    candidates = int(config.get("sampling", {}).get("candidates", 4))
    steps = int(config.get("sampling", {}).get("steps", 30))
    for batch_index, batch in enumerate(loader):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        sampled = sample_candidates(
            model,
            batch,
            sde,
            candidates=candidates,
            steps=steps,
            seed=seed + batch_index,
        )
        evidence = _evidence(sampled, batch["initial_state"])
        prediction = state_to_matrices(sampled.float())
        target = batch["target_matrix"][:, None].expand_as(prediction)
        error = geodesic_distance(prediction, target)
        valid = (batch["target_rotation_valid"] & batch["frame_valid"][..., None])[
            :, None
        ]
        error = (error * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)
        feature_rows.append(evidence.cpu().numpy().astype(np.float32))
        error_rows.append(error.cpu().numpy().astype(np.float32))
        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "selector_feature_progress",
                        "batches": batch_index + 1,
                        "samples": sum(len(value) for value in feature_rows),
                    }
                ),
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            evidence_features=np.concatenate(feature_rows),
            target_errors=np.concatenate(error_rows),
            posterior_sha256=np.asarray(sha256_file(args.checkpoint)),
            config_sha256=np.asarray(sha256_file(args.config)),
            seed=np.asarray(seed),
        )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
