import cv2

from wingjournal.templates import (
    build_legend_pdf,
    build_writing_sheet,
    compute_layout,
    render_writing_sheet,
)
from wingjournal.vision.aruco import detect_markers


def test_layout_marker_positions_inside_page():
    layout = compute_layout(paper="letter", dpi=200)
    for x, y in layout.marker_xy.values():
        assert 0 <= x < layout.width_px
        assert 0 <= y < layout.height_px


def test_writing_sheet_markers_are_detectable():
    sheet = render_writing_sheet(compute_layout(dpi=200))
    ids = {m.marker_id for m in detect_markers(sheet)}
    assert {0, 1, 2, 3} <= ids                      # the four page corners
    assert {20, 21, 22, 23, 24, 25, 26} <= ids      # the seven field anchors


def test_ruled_sheet_still_detectable():
    sheet = render_writing_sheet(compute_layout(dpi=200), ruled=True)
    assert {0, 1, 2, 3} <= {m.marker_id for m in detect_markers(sheet)}


def test_build_writing_sheet_pdf_multipage(tmp_path):
    out = build_writing_sheet(tmp_path / "s.pdf", pages=3, dpi=150)
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    # 3 content pages + 1 /Pages node
    assert data.count(b"/Type /Page") == 4


def test_build_writing_sheet_png(tmp_path):
    out = build_writing_sheet(tmp_path / "s.png", dpi=150)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    img = cv2.imread(str(out))
    assert img is not None
    assert {0, 1, 2, 3} <= {m.marker_id for m in detect_markers(img)}


def test_build_writing_sheet_png_rejects_multipage(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        build_writing_sheet(tmp_path / "s.png", pages=2)


def test_build_legend_pdf(tmp_path):
    out = build_legend_pdf(tmp_path / "legend.pdf", dpi=120)
    assert out.read_bytes()[:5] == b"%PDF-"


def test_a4_paper(tmp_path):
    out = build_writing_sheet(tmp_path / "a4.pdf", paper="a4", dpi=120)
    assert out.is_file()
