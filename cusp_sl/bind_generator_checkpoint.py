"""Create an append-only generator checkpoint with cryptographic Q binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_payload(payload: dict, *, source_sha256: str, q_sha256: str) -> dict:
    bound = dict(payload)
    bound["artifact_role"] = "posthoc_provenance_bound_generator_v1"
    bound["source_checkpoint_sha256"] = source_sha256
    bound["reliability_checkpoint_sha256"] = q_sha256
    return bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reliability-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    q_payload = torch.load(
        args.reliability_checkpoint, map_location="cpu", weights_only=False
    )
    if payload.get("config_sha256") != q_payload.get("config_sha256"):
        raise ValueError("Generator and reliability checkpoints use different configs")
    recorded = Path(str(payload.get("reliability_checkpoint", ""))).resolve()
    if recorded != args.reliability_checkpoint.resolve():
        raise ValueError(
            "Generator's recorded reliability path differs from the binding input"
        )
    source_hash = sha256(args.input)
    q_hash = sha256(args.reliability_checkpoint)
    bound = bind_payload(payload, source_sha256=source_hash, q_sha256=q_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bound, args.output)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_checkpoint_sha256": source_hash,
        "reliability_checkpoint_sha256": q_hash,
        "step": int(payload["step"]),
        "model_kind": payload.get("model_kind"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
