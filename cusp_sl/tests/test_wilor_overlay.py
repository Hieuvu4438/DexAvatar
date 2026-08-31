import numpy as np

from cusp_sl.audit_wilor_overlays import project_full_image


def test_full_image_projection_uses_camera_translation_and_center():
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, -0.5, 0.0]], np.float32)
    pixels, depth = project_full_image(
        points,
        np.asarray([0.0, 0.0, 2.0], np.float32),
        100.0,
        np.asarray([640.0, 480.0], np.float32),
    )
    np.testing.assert_allclose(depth, [2.0, 2.0])
    np.testing.assert_allclose(pixels, [[320.0, 240.0], [370.0, 215.0]])
