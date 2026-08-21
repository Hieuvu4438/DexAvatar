import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "extract_nlf_observations.py"
SPEC = importlib.util.spec_from_file_location("extract_nlf_observations", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_detection_index_uses_image_area_and_confidence():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0, 0.9],
            [0.0, 0.0, 200.0, 150.0, 0.8],
        ]
    )
    assert MODULE.detection_index(boxes) == 1


def test_detection_index_rejects_empty_boxes():
    try:
        MODULE.detection_index(torch.empty(0, 5))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty detections must fail")
