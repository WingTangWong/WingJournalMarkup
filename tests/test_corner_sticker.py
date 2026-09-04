import cv2
import numpy as np

from wingjournal.templates.corner_sticker import build_sticker_sheet, render_corner_sticker
from wingjournal.vision.corner_sticker import (
    detect_corner_stickers,
    estimate_page_size,
    sticker_quad,
)


def _sticker_page(dpi: int = 150, paper=(216, 279), inset_mm: float = 1.0):
    pxmm = dpi / 25.4
    W, H = int(paper[0] * pxmm), int(paper[1] * pxmm)
    page = np.full((H, W, 3), 255, np.uint8)
    st = render_corner_sticker(sticker_mm=26, dpi=dpi)
    s = st.shape[0]
    off = int(inset_mm * pxmm)
    rots = [
        (st, off, off),
        (cv2.rotate(st, cv2.ROTATE_90_CLOCKWISE), W - s - off, off),
        (cv2.rotate(st, cv2.ROTATE_180), W - s - off, H - s - off),
        (cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE), off, H - s - off),
    ]
    for img, x, y in rots:
        page[y:y + img.shape[0], x:x + img.shape[1]] = img
    return page, (W, H)


def test_detects_four_stickers_with_roles():
    page, _ = _sticker_page()
    st = detect_corner_stickers(page)
    assert len(st) == 4
    assert {s.inferred_role for s in st} == {"TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"}
    assert all(s.bracket_found for s in st)


def test_sticker_quad_spans_the_page():
    page, (W, H) = _sticker_page(inset_mm=1.0)
    quad = sticker_quad(detect_corner_stickers(page))
    assert quad is not None
    tl, tr, br, bl = quad
    assert tl[0] < 0.15 * W and tl[1] < 0.15 * H
    assert br[0] > 0.85 * W and br[1] > 0.85 * H


def test_page_size_estimate_matches_letter():
    page, _ = _sticker_page(paper=(216, 279), inset_mm=0.0)
    est = estimate_page_size(detect_corner_stickers(page))
    assert est is not None
    assert est.best_match == "letter"
    assert est.match_error_mm < 20
    assert 200 < est.width_mm < 230 and 265 < est.height_mm < 292


def test_a4_page_estimated_as_a4():
    page, _ = _sticker_page(paper=(210, 297), inset_mm=0.0)
    est = estimate_page_size(detect_corner_stickers(page))
    assert est is not None and est.best_match == "a4"


def test_boundary_uses_corner_stickers():
    from wingjournal.pipeline import ingest_image

    page, _ = _sticker_page()
    cv2.rectangle(page, (200, 120), (page.shape[1] - 200, 220), (0, 0, 0), 3)
    r = ingest_image("s", page)
    assert r.capture.page_boundary_method == "corner_stickers"
    assert r.capture.page_size_estimate is not None
    assert r.capture.page_size_estimate["best_match"] == "letter"


def test_no_stickers_no_estimate():
    blank = np.full((900, 700, 3), 255, np.uint8)
    assert detect_corner_stickers(blank) == []
    assert estimate_page_size([]) is None


def test_make_stickers_sheet(tmp_path):
    out = build_sticker_sheet(tmp_path / "s.pdf", count=6)
    assert out.read_bytes()[:5] == b"%PDF-"
    png = build_sticker_sheet(tmp_path / "s.png", count=6)
    img = cv2.imread(str(png))
    # the printed stickers are detectable straight off the sheet
    assert len(detect_corner_stickers(img)) >= 4
