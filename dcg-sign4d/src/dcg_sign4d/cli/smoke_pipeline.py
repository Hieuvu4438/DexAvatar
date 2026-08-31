from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.inference.artifacts import write_prediction_artifact
from dcg_sign4d.inference.provenance import build_run_identity
from dcg_sign4d.synthetic import make_observations, make_state
from dcg_sign4d.synthetic_pipeline import build_smoke_reconstructor
from dcg_sign4d.utils.hashing import file_sha256, tensor_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the development-only synthetic pipeline")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config["experiment"].get("development_only") or config_path.name != "smoke.yaml":
        raise ValueError("synthetic pipeline requires configs/smoke.yaml development marker")
    state = make_state(time=8)
    observations = make_observations(time=8)
    patch_map_path = Path(config["geometry"]["patch_map"])
    patch_map = PatchMap.load(patch_map_path)
    started_at = datetime.now(UTC).isoformat()
    reconstructor = build_smoke_reconstructor(
        state,
        patch_map,
        rounds=config["inference"]["rounds"],
        diffusion_steps=config["inference"]["diffusion_steps"],
        num_hypotheses=config["inference"]["num_hypotheses"],
        seed=config["experiment"]["seed"],
    )
    hypotheses = reconstructor.reconstruct(state, observations)
    ended_at = datetime.now(UTC).isoformat()
    dependency_manifest = yaml.safe_load(
        Path("third_party/manifest.yaml").read_text(encoding="utf-8")
    )
    run_identity = build_run_identity(
        scope_root=Path.cwd(),
        config_path=config_path,
        manifest_path=config["data"]["manifest"],
        dependency_commits={
            item["name"]: item["commit"] for item in dependency_manifest["repositories"]
        },
        checkpoint_sha256={},
        sampler={
            "diffusion_steps": config["inference"]["diffusion_steps"],
            "rounds": config["inference"]["rounds"],
            "num_hypotheses": config["inference"]["num_hypotheses"],
        },
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        peak_memory_bytes=0,
        frame_count=state.root_rot6d.shape[1],
        execution_device=str(state.root_rot6d.device),
        development_only=True,
    )
    artifact = write_prediction_artifact(
        args.output,
        "synthetic_smoke",
        (state, {"backend": "synthetic_fixture"}),
        hypotheses,
        {**run_identity, "patch_map_sha256": patch_map.content_hash},
        input_manifest={
            "clip_id": "synthetic_smoke",
            "manifest_sha256": file_sha256(config["data"]["manifest"]),
        },
        observation_hashes={
            "keypoints_2d": tensor_sha256(observations.keypoints_2d),
            "keypoint_reliability": tensor_sha256(observations.keypoint_reliability),
            "keypoint_valid": tensor_sha256(observations.keypoint_valid),
            "frame_valid": tensor_sha256(observations.frame_valid),
        },
        ranker_config={
            "weights": {"observation": 1, "contact": 1, "event": 1, "motion": 1},
            "fit_split": "synthetic_validation_fixture",
            "development_only": True,
        },
    )
    print(json.dumps({"artifact": str(artifact), "hypotheses": len(hypotheses)}, indent=2))


if __name__ == "__main__":
    main()
