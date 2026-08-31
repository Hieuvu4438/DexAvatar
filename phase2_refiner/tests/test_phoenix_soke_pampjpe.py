import json

import pytest
import torch

from phase2_refiner.evaluate_phoenix_soke_pampjpe import _rigid_align_soke, _write


def test_soke_procrustes_recovers_similarity_transform() -> None:
    generator = torch.Generator().manual_seed(42)
    prediction = torch.randn(5, 21, 3, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    target = 1.7 * (prediction @ rotation.T) + torch.tensor([0.4, -1.2, 2.0])
    aligned = _rigid_align_soke(prediction, target)
    torch.testing.assert_close(aligned, target, atol=1e-5, rtol=1e-5)


def test_evaluation_json_is_published_atomically(tmp_path) -> None:
    output = tmp_path / "evaluation.json"
    _write(output, {"mode": "test"})
    assert json.loads(output.read_text()) == {"mode": "test"}
    assert list(tmp_path.glob(".evaluation.json.tmp.*")) == []
    with pytest.raises(FileExistsError):
        _write(output, {"mode": "overwrite"})
