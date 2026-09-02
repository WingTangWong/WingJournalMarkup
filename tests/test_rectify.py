import cv2
import numpy as np

from wingjournal.vision.rectify import TARGET_LONG_PX, output_size, rectify


def test_output_size_is_fixed_and_aspect_clamped():
    # a portrait-ish quad
    quad = np.array([[0, 0], [400, 0], [400, 560], [0, 560]], dtype=np.float32)
    w, h = output_size(quad)
    assert max(w, h) == TARGET_LONG_PX
    assert h > w
    assert 1.15 <= h / w <= 1.6

    # an absurd aspect gets clamped, not passed through
    wide = np.array([[0, 0], [4000, 0], [4000, 100], [0, 100]], dtype=np.float32)
    w2, h2 = output_size(wide)
    assert max(w2, h2) == TARGET_LONG_PX
    assert 1.15 <= max(w2, h2) / min(w2, h2) <= 1.6


def test_repeated_captures_normalize_to_the_same_size():
    src = np.zeros((6, 6, 3), np.uint8)
    near = np.array([[10, 10], [210, 12], [208, 300], [12, 298]], dtype=np.float32)
    far = near * 0.4 + 5  # same page, smaller in frame
    n1, _ = rectify(src, near)
    n2, _ = rectify(src, far)
    assert n1.shape == n2.shape


def test_size_override_still_works():
    quad = np.array([[0, 0], [100, 0], [100, 130], [0, 130]], dtype=np.float32)
    out, h = rectify(np.zeros((5, 5, 3), np.uint8), quad, size=(80, 100))
    assert out.shape[:2] == (100, 80)
    assert h.shape == (3, 3)


def test_homography_maps_quad_to_the_fixed_frame():
    quad = np.array([[5, 7], [190, 3], [195, 250], [8, 255]], dtype=np.float32)
    _, homography = rectify(np.zeros((300, 300, 3), np.uint8), quad)
    w, h = output_size(quad)
    mapped = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), homography).reshape(-1, 2)
    expected = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    np.testing.assert_allclose(mapped, expected, atol=1.0)
