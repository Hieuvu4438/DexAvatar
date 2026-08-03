from __future__ import annotations

import torch
import pytest

from phase3_posterior.data.cache_schema import reject_forbidden_path
from phase3_posterior.data.corruptions import sample_conditioning_mask
from phase3_posterior.data.evidence_split import conditioning_mask, evidence_mask


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
