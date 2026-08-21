import torch

from signal4d.data.cache import ObservationBatch
from signal4d.io.predictions import PredictionArtifact


def test_cache_hash_and_shape_roundtrip(tmp_path) -> None:
    batch = ObservationBatch(
        frame_ids=torch.arange(4),
        joints_3d=torch.zeros(4, 2, 3, 3),
        valid_3d=torch.ones(4, 2, 3, dtype=torch.bool),
        features=torch.zeros(4, 2, 3, 4),
    )
    batch.save(tmp_path, {"clip_id": "x"})
    recovered, metadata = ObservationBatch.load(tmp_path)
    assert recovered.joints_3d.shape == (4, 2, 3, 3)
    assert metadata["clip_id"] == "x"


def test_prediction_keeps_pose_when_abstaining(tmp_path) -> None:
    prediction = PredictionArtifact(
        frame_ids=torch.arange(3),
        joints_3d=torch.randn(3, 4, 3),
        rotations=None,
        translation=torch.zeros(3, 3),
        vertices=None,
        risk_score=torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.2, 0.1], [0.1, 0.1, 0.1]]),
        abstain=torch.tensor([[False, False, False], [True, False, False], [False, False, False]]),
        uncertainty=torch.ones(3, 4) * 0.01,
    )
    prediction.save(tmp_path, {"status": "abstained_high_risk"})
    recovered, _ = PredictionArtifact.load(tmp_path)
    assert recovered.joints_3d.shape == (3, 4, 3)
    assert recovered.abstain[1, 0]
