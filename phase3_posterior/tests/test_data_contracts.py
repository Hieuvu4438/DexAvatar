from __future__ import annotations

import torch
import pytest

from phase3_posterior.data.cache_schema import reject_forbidden_path
from phase3_posterior.data.corruptions import sample_conditioning_mask
from phase3_posterior.data.evidence_split import conditioning_mask, evidence_mask
from phase3_posterior.data.build_phase3_index import _resolve_signer
from phase3_posterior.config import validate_config


def test_forbidden_author_and_sgnify_paths() -> None:
    with pytest.raises(ValueError):
        reject_forbidden_path("/repo/data/evaluation_from_author/train.npz")
    with pytest.raises(ValueError):
        reject_forbidden_path("/repo/data/smplx_gt/target.npz")


def test_mask_never_adds_invalid_supervision() -> None:
    valid = torch.ones(16, 51, dtype=torch.bool)
    valid[0, 0] = False
    result = sample_conditioning_mask(
        valid, torch.Generator().manual_seed(42), mode="left_hand"
    )
    assert not result.conditioning[0, 0]
    assert (~result.conditioning[:, 21:36]).any()


def test_conditioning_and_evidence_are_disjoint() -> None:
    valid = torch.ones(12, 51, dtype=torch.bool)
    evidence = evidence_mask(valid, "clip", fold=0)
    conditioning = conditioning_mask(valid, evidence)
    assert not (conditioning & evidence).any()
    assert torch.equal(conditioning | evidence, valid)


def test_how2sign_signer_resolver_uses_terminal_filename_identity() -> None:
    source = {"signer_resolver": "how2sign_filename_v1"}
    metadata = {"source_clip": "g1xdqxCZxTg_14-3-rgb_front"}
    assert _resolve_signer(source, metadata) == "how2sign_signer_03"


def test_geometry_only_fallback_rejects_contact_loss() -> None:
    config = {
        "data": {},
        "model": {"num_joints": 51, "max_frames": 8, "contact_energy_enabled": False},
        "diffusion": {"beta_min": 0.1, "beta_max": 20.0, "eps": 0.001},
        "training": {"workers": 0, "gradient_accumulation": 1},
        "loss": {"contact": 0.25, "persistence": 0.0},
        "fallback": {
            "mode": "geometry_only",
            "contact_energy_enabled": False,
            "force_coupling_enabled": False,
            "persistence_constraints_enabled": False,
        },
    }
    with pytest.raises(ValueError, match="loss.contact=0"):
        validate_config(config)
