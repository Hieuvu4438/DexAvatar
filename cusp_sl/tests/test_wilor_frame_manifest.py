import pytest

from cusp_sl.prepare_wilor_frame_manifest import cache_image_size_order


def test_cache_image_size_order_accepts_both_conventions():
    assert cache_image_size_order((514, 300), (514, 300)) == "width_height"
    assert cache_image_size_order((300, 514), (514, 300)) == "height_width"


def test_cache_image_size_order_rejects_inconsistent_dimensions():
    with pytest.raises(ValueError, match="inconsistent"):
        cache_image_size_order((256, 256), (514, 300))
