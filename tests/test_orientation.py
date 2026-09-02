import cv2
import numpy as np
import pytest

from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.orientation import resolve_orientation, rotate_upright
from wingjournal.vision.preprocess import preprocess
from wingjournal.vision.synthetic import make_page, warp_page

_ROT = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@pytest.mark.parametrize("rot_cw", [0, 90, 180, 270])
def test_orientation_from_marker_ids(rot_cw):
    page = make_page()
    if _ROT[rot_cw] is not None:
        page = cv2.rotate(page, _ROT[rot_cw])
    scene, _ = warp_page(page, seed=5)
    pre = preprocess(scene)
    markers = detect_markers(pre.gray)

    o = resolve_orientation(pre, markers)
    assert o.method == "aruco_ids"
    assert o.degrees == (360 - rot_cw) % 360
    assert o.flip_ambiguous is False


def test_rotate_upright_roundtrip():
    img = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)
    back = rotate_upright(rotate_upright(img, 90), 270)
    np.testing.assert_array_equal(img, back)


def test_rotate_upright_rejects_non_multiple():
    with pytest.raises(ValueError):
        rotate_upright(np.zeros((3, 3), np.uint8), 45)


def test_text_baseline_fallback_no_markers():
    blank = np.full((400, 300), 255, np.uint8)
    blank[100:110, 40:260] = 0  # one horizontal ink band
    blank[150:160, 40:260] = 0
    pre = preprocess(blank)
    o = resolve_orientation(pre, [])
    assert o.method in {"text_baseline", "assumed"}
    # marker-free resolution can never rule out a 180 flip
    assert o.flip_ambiguous is True
