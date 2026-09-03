import cv2

from wingjournal.templates.geometry import compute_layout
from wingjournal.templates.writing_sheet import render_writing_sheet
from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.registration import RegistrationMark
from wingjournal.vision.sharpness import assess, laplacian_variance


def _sheet(dpi: int = 200):
    return render_writing_sheet(compute_layout(dpi=dpi))


def test_laplacian_variance_drops_with_blur():
    sheet = _sheet()
    v0 = laplacian_variance(sheet)
    v1 = laplacian_variance(cv2.GaussianBlur(sheet, (0, 0), 3))
    assert v0 > v1 * 3


def test_sharp_page_scores_high_and_is_not_blurry():
    sheet = _sheet()
    marks = [RegistrationMark([100, 100], 20, acutance=0.9)]
    rep = assess(sheet, detect_markers(sheet), marks)
    assert rep.score > 0.6
    assert not rep.blurry


def test_blurred_page_is_flagged():
    sheet = cv2.GaussianBlur(_sheet(), (0, 0), 4)
    marks = [RegistrationMark([100, 100], 20, acutance=0.1)]
    rep = assess(sheet, detect_markers(sheet), marks)
    assert rep.score < 0.45
    assert rep.blurry


def test_one_soft_probe_flags_the_capture():
    sheet = _sheet()
    marks = [
        RegistrationMark([100, 100], 20, acutance=0.8),
        RegistrationMark([200, 100], 20, acutance=0.05),  # this one is mush
    ]
    rep = assess(sheet, [], marks)
    assert rep.blurry
    assert [p.sharp for p in rep.probes] == [True, False]


def test_report_summary_names_the_soft_probes():
    rep = assess(_sheet(), [], [RegistrationMark([50, 50], 20, acutance=0.02)])
    assert "registration:0" in rep.summary()
