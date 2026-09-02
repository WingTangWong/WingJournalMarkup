from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.fiducial_candidates import find_square_candidates
from wingjournal.vision.preprocess import preprocess
from wingjournal.vision.synthetic import make_page


def test_blank_corner_squares_are_catalogued():
    page = make_page(blank_roles=("TOP_LEFT", "BOTTOM_RIGHT"))
    pre = preprocess(page)
    markers = detect_markers(pre.gray)
    assert {m.marker_id for m in markers} == {1, 3}  # TR, BL still decode

    squares = find_square_candidates(pre, exclude=markers)
    roles = {s.inferred_role for s in squares}
    assert "TOP_LEFT" in roles
    assert "BOTTOM_RIGHT" in roles
    assert all(0.0 <= s.confidence <= 1.0 and not s.decoded for s in squares)


def test_no_false_squares_on_plain_page():
    page = make_page()
    pre = preprocess(page)
    markers = detect_markers(pre.gray)
    squares = find_square_candidates(pre, exclude=markers)
    # the metadata block and node box are not square enough to pass
    assert len(squares) <= 1
