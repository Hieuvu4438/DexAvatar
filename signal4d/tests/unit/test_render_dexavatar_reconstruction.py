import numpy as np
import pytest

from signal4d.cli.render_dexavatar_reconstruction import _composite_rgba


def test_composite_rgba_uses_renderer_alpha_and_requested_opacity() -> None:
    background = np.full((1, 2, 3), 100, dtype=np.uint8)
    rendered = np.array([[[200, 50, 0, 255], [200, 50, 0, 0]]], dtype=np.uint8)

    result = _composite_rgba(background, rendered, 0.5)

    np.testing.assert_array_equal(result[0, 0], np.array([150, 75, 50]))
    np.testing.assert_array_equal(result[0, 1], np.array([100, 100, 100]))


def test_composite_rgba_rejects_shape_and_opacity_errors() -> None:
    background = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        _composite_rgba(background, np.zeros((1, 2, 4), dtype=np.uint8), 0.9)
    with pytest.raises(ValueError, match="opacity"):
        _composite_rgba(background, np.zeros((2, 2, 4), dtype=np.uint8), 1.1)
