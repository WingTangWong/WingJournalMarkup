import numpy as np

from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.boundary import assign_roles_greedy, best_roles, order_points


def test_order_points_canonical():
    pts = np.array([[10, 10], [90, 20], [80, 95], [5, 88]], dtype=np.float32)
    shuffled = pts[[2, 0, 3, 1]]
    ordered = order_points(shuffled)
    np.testing.assert_allclose(ordered, pts, atol=1e-4)


def test_detect_four_markers_on_flat_page(flat_page):
    markers = detect_markers(flat_page)
    assert {m.marker_id for m in markers} == {0, 1, 2, 3}


def test_marker_roles_flat_page(flat_page):
    markers = detect_markers(flat_page)
    for roles in (best_roles(markers), assign_roles_greedy(markers)):
        assert roles["TOP_LEFT"].marker_id == 0
        assert roles["TOP_RIGHT"].marker_id == 1
        assert roles["BOTTOM_RIGHT"].marker_id == 2
        assert roles["BOTTOM_LEFT"].marker_id == 3


def test_detect_markers_under_perspective(warped_scene):
    scene, _quad = warped_scene
    markers = detect_markers(scene)
    assert len(markers) == 4
