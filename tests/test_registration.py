import cv2
import numpy as np

from wingjournal.templates.geometry import compute_layout
from wingjournal.templates.writing_sheet import render_writing_sheet
from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.registration import (
    RegistrationMark,
    detect_registration_marks,
    marks_to_quad,
)


def _sheet(dpi: int = 200):
    layout = compute_layout(dpi=dpi)
    return render_writing_sheet(layout), layout


def _marker_boxes(img):
    out = []
    for m in detect_markers(img):
        x, y, w, h = cv2.boundingRect(np.asarray(m.corners, dtype=np.float32))
        out.append((x - 5, y - 5, w + 10, h + 10))
    return out


def test_finds_four_marks_on_the_sheet():
    sheet, layout = _sheet()
    band = (0, 0, sheet.shape[1], int(sheet.shape[0] * 0.3))
    marks = detect_registration_marks(sheet, roi=band, exclude=_marker_boxes(sheet))
    assert len(marks) == 4
    assert all(m.rings >= 2 for m in marks)
    assert all(m.acutance > 0.6 for m in marks)  # crisp render


def test_marks_match_the_layout_corners():
    sheet, layout = _sheet()
    band = (0, 0, sheet.shape[1], int(sheet.shape[0] * 0.3))
    marks = detect_registration_marks(sheet, roi=band, exclude=_marker_boxes(sheet))
    quad = marks_to_quad(marks)
    assert quad is not None
    got = {(round(x / 20), round(y / 20)) for x, y in quad}
    want = {(round(x / 20), round(y / 20)) for x, y in layout.registration_xy}
    assert got == want


def test_acutance_drops_when_blurred():
    sheet, _ = _sheet()
    band = (0, 0, sheet.shape[1], int(sheet.shape[0] * 0.3))
    excl = _marker_boxes(sheet)
    sharp = detect_registration_marks(sheet, roi=band, exclude=excl)
    soft = detect_registration_marks(cv2.GaussianBlur(sheet, (0, 0), 3.0), roi=band, exclude=excl)
    assert np.mean([m.acutance for m in sharp]) > 0.6
    if soft:  # blur can also cost a detection; whatever survives is softer
        assert np.mean([m.acutance for m in soft]) < np.mean([m.acutance for m in sharp])


def test_marks_to_quad_needs_exactly_four():
    assert marks_to_quad([]) is None
    three = [RegistrationMark([0, 0], 10), RegistrationMark([10, 0], 10),
             RegistrationMark([0, 10], 10)]
    assert marks_to_quad(three) is None


def test_excluded_regions_are_ignored():
    sheet, _ = _sheet()
    band = (0, 0, sheet.shape[1], int(sheet.shape[0] * 0.3))
    # exclude the whole band -> nothing
    whole = [(0, 0, sheet.shape[1], sheet.shape[0])]
    assert detect_registration_marks(sheet, roi=band, exclude=whole) == []
