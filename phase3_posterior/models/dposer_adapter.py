"""Fail-closed adapter for an optional frozen DPoser-X whole-body score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from phase3_posterior.provenance import sha256_file


class SpatialScore(Protocol):
    def __call__(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor: ...


class ZeroSpatialPrior(nn.Module):
    """Explicit from-scratch route; never represented as a pretrained prior."""

    pretrained = False

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(state)


class FrozenSpatialPrior(nn.Module):
    """Wrap a validated 51x6 score model and prevent accidental fine-tuning."""

    pretrained = True

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenSpatialPrior":
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            score = self.model(state, time)
        if score.shape != state.shape or not torch.isfinite(score).all():
            raise ValueError(
                "Frozen DPoser score violates the Phase 3 (B,T,51,6) contract"
            )
        return score


def audit_dposer_contract(contract_path: str | Path) -> dict[str, object]:
    """Validate files, hashes, representation, joint count, and license metadata."""
    path = Path(contract_path)
    with path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    checks: dict[str, bool] = {}
    for item in contract.get("files", []):
        source = Path(item["path"])
        checks[f"exists:{source}"] = source.is_file()
        if source.is_file():
            checks[f"hash:{source}"] = sha256_file(source) == item["sha256"]
    checks["whole_body_51_joints"] = int(contract.get("joint_count", 0)) == 51
    checks["rotation_6d"] = contract.get("representation") == "rotation_6d"
    checks["sub_vp_sde"] = contract.get("sde") == "sub_vp"
    checks["normalizer_recorded"] = bool(contract.get("normalizer_sha256"))
    checks["license_recorded"] = bool(contract.get("license_id"))
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "contract": str(path.resolve()),
        "route_on_failure": "teacher_distillation_or_from_scratch",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_dposer_contract(args.contract)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
