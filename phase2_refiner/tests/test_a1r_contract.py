import numpy as np

from phase2_refiner.data.a1r_contract import (
    infer_fitting_contract,
    write_fitting_contract,
)
from phase2_refiner.tests.test_cache import make_clip


def _observed_clip():
    clip = make_clip(5)
    clip.keypoint_valid[:] = True
    clip.validate()
    clip.track_valid[:] = True
    return clip


def test_a1r_contract_infers_one_hand_without_sign_label(tmp_path):
    clip = _observed_clip()
    clip.keypoints_2d[:, 21:36, 0] = np.arange(5)[:, None] * 0.01
    contract = infer_fitting_contract(clip, "portable_clip")
    assert contract.sign_class == "0"
    assert contract.active_side == "left"
    paths = write_fitting_contract(tmp_path / "contract", contract)
    assert paths["sign_class"].read_text() == "portable_clip 0\n"
    assert '"portable_clip"' in paths["sign_segment"].read_text()


def test_a1r_contract_infers_two_active_hands():
    clip = _observed_clip()
    displacement = np.arange(5, dtype=np.float32)[:, None] * 0.01
    clip.keypoints_2d[:, 21:36, 0] = displacement
    clip.keypoints_2d[:, 36:51, 1] = displacement
    contract = infer_fitting_contract(clip, "portable_clip")
    assert contract.sign_class == "1"
    assert contract.active_side == "both"
