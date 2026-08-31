import numpy as np

from signpccx.optimization.camera import weighted_huber_line

from signpccx.geometry.camera import affine_2x3_to_homogeneous, project_opencv, transform_xy


def test_weighted_huber_camera_line_rejects_large_outlier():
    coordinate = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 0.25])
    pixels = 500.0 * coordinate + 256.0
    pixels[-1] += 1000.0
    slope, center = weighted_huber_line(coordinate, pixels, np.ones(6), delta_px=2.0, iterations=20)
    assert abs(slope - 500.0) < 5.0
    assert abs(center - 256.0) < 5.0
from signpccx.geometry.handedness import unmirror_left_axis_angle


def test_crop_roundtrip():
    affine = np.asarray([[1.25, 0.0, -30.0], [0.0, 1.25, 12.0]], dtype=np.float32)
    full_to_crop = affine_2x3_to_homogeneous(affine)
    crop_to_full = np.linalg.inv(full_to_crop)
    points = np.asarray([[0.0, 0.0], [100.25, 200.75], [511.0, 383.0]], dtype=np.float32)
    recovered = transform_xy(transform_xy(points, full_to_crop), crop_to_full)
    assert np.max(np.abs(recovered - points)) < 1e-4


def test_projection():
    intrinsic = np.asarray([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.asarray([[1.0, 2.0, 10.0], [-1.0, 0.5, 5.0]])
    expected = np.asarray([[420.0, 440.0], [120.0, 340.0]])
    np.testing.assert_allclose(project_opencv(points, intrinsic), expected, atol=1e-8)


def test_left_axis_angle_unmirror_is_involution():
    pose = np.arange(45, dtype=np.float32).reshape(15, 3)
    np.testing.assert_array_equal(unmirror_left_axis_angle(unmirror_left_axis_angle(pose)), pose)
