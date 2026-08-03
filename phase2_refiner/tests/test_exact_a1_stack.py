import numpy as np
import pytest

from phase2_refiner.data.run_exact_a1_stack import (
    REQUIRED_RESULT_SHAPES,
    parse_video_frame_reference,
    validate_exact_result_payload,
)


def valid_payload():
    payload = {
        key: np.zeros((1, size), dtype=np.float32)
        for key, size in REQUIRED_RESULT_SHAPES.items()
    }
    payload["K"] = np.eye(3, dtype=np.float32)
    return payload


def test_parse_video_frame_reference(tmp_path):
    video = tmp_path / "source.mp4"
    video.touch()
    path, frame = parse_video_frame_reference(f"{video}#frame=17")
    assert path == video.resolve()
    assert frame == 17


@pytest.mark.parametrize("reference", ["video.mp4", "video.mp4#frame=x"])
def test_parse_video_frame_reference_rejects_invalid(reference):
    with pytest.raises(ValueError):
        parse_video_frame_reference(reference)


def test_validate_exact_result_payload_accepts_complete_schema():
    validate_exact_result_payload(valid_payload(), "memory")


@pytest.mark.parametrize("key", ["K", "transl", "right_hand_pose"])
def test_validate_exact_result_payload_rejects_missing_key(key):
    payload = valid_payload()
    payload.pop(key)
    with pytest.raises(ValueError, match=key):
        validate_exact_result_payload(payload, "memory")


def test_validate_exact_result_payload_rejects_invalid_camera():
    payload = valid_payload()
    payload["K"] = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="K shape"):
        validate_exact_result_payload(payload, "memory")
