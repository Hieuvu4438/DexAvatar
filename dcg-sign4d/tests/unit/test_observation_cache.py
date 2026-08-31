import torch

from dcg_sign4d.observations.cache import ObservationCache
from dcg_sign4d.observations.schema import ObservationBatch


def observations():
    keypoints = torch.tensor([[[[10.0, 20.0], [float("nan"), float("nan")]]]])
    return ObservationBatch(
        keypoints_2d=keypoints,
        keypoint_reliability=torch.tensor([[[0.8, 0.0]]]),
        keypoint_valid=torch.tensor([[[True, False]]]),
        frame_valid=torch.tensor([[True]]),
        metadata=({"clip_id": "fixture"},),
    )


def test_missing_cue_is_masked_not_fabricated():
    observations().validate()


def test_reliability_range_is_enforced():
    value = observations()
    object.__setattr__(value, "keypoint_reliability", torch.tensor([[[1.1, 0.0]]]))
    try:
        value.validate()
    except ValueError as exc:
        assert "[0, 1]" in str(exc)
    else:
        raise AssertionError("invalid reliability accepted")


def test_cache_identity_and_round_trip(tmp_path):
    cache = ObservationCache(tmp_path)
    kwargs = {
        "video_hash": "v",
        "extractor": {"name": "fixture", "version": "1", "checkpoint": "c"},
        "preprocessing": {"size": 256},
        "calibration_hash": "t",
    }
    first = cache.identity(**kwargs)
    assert first == cache.identity(**kwargs)
    assert first != cache.identity(**{**kwargs, "calibration_hash": "other"})
    cache.save(first, observations())
    restored = cache.load(first)
    assert torch.equal(restored.keypoint_valid, observations().keypoint_valid)
    assert torch.allclose(
        restored.keypoints_2d[restored.keypoint_valid],
        observations().keypoints_2d[observations().keypoint_valid],
    )
    try:
        cache.save(first, observations())
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable cache was overwritten")


def test_cache_tensor_tamper_is_detected(tmp_path):
    cache = ObservationCache(tmp_path)
    cache.save("fixture", observations())
    target = tmp_path / "fixture" / "observations.npz"
    target.write_bytes(target.read_bytes() + b"tamper")
    try:
        cache.load("fixture")
    except ValueError as exc:
        assert "tensor hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered observation cache was accepted")
