import random

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


def _rotate_expand(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def _misrotated_page(seed: int, warp: bool = False) -> np.ndarray:
    """A page whose stickers were stuck on rotated 10-95 deg the wrong way."""
    rng = random.Random(seed)
    pxmm = 200 / 25.4
    W, H = int(216 * pxmm), int(279 * pxmm)
    page = np.full((H, W, 3), 255, np.uint8)
    base = render_corner_sticker(sticker_mm=26, dpi=200)
    for role, correct in (("TL", 0), ("TR", -90), ("BR", 180), ("BL", 90)):
        wrong = rng.uniform(10, 95) * rng.choice([-1, 1])
        st = _rotate_expand(base, correct + wrong)
        sh, sw = st.shape[:2]
        off = 30
        x, y = {
            "TL": (off, off), "TR": (W - sw - off, off),
            "BR": (W - sw - off, H - sh - off), "BL": (off, H - sh - off),
        }[role]
        roi = page[y:y + sh, x:x + sw]
        mask = (st < 250).any(axis=2)
        roi[mask] = st[mask]
    cv2.putText(page, "ROTATED STICKERS", (int(W * 0.2), int(H * 0.4)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(page, "the quick brown fox", (int(W * 0.15), int(H * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    if warp:
        from wingjournal.vision.synthetic import warp_page
        page, _ = warp_page(page, seed=seed)
    return page


def test_roles_survive_wrong_way_rotation():
    # the corner role comes from the constellation, not the sticker graphic
    for seed in range(12):
        st = detect_corner_stickers(_misrotated_page(seed))
        assert len(st) == 4
        assert sorted(s.inferred_role for s in st) == [
            "BOTTOM_LEFT", "BOTTOM_RIGHT", "TOP_LEFT", "TOP_RIGHT",
        ], f"seed {seed}: {[s.inferred_role for s in st]}"
        assert sticker_quad(st) is not None


def test_capture_upright_with_misrotated_stickers():
    from wingjournal.pipeline import ingest_image

    for seed in range(12):
        r = ingest_image("s", _misrotated_page(seed))
        assert r.capture.page_boundary_method == "corner_stickers", f"seed {seed}"
        assert r.capture.orientation_degrees in (0, 180), f"seed {seed}"
        ni = r.normalized_image
        assert ni.shape[0] > ni.shape[1], f"seed {seed}: not portrait {ni.shape}"


def test_capture_survives_misrotation_plus_perspective():
    from wingjournal.pipeline import ingest_image

    ok = 0
    for seed in range(10):
        r = ingest_image("w", _misrotated_page(seed, warp=True))
        ni = r.normalized_image
        if (r.capture.page_boundary_method == "corner_stickers"
                and r.capture.orientation_degrees in (0, 180)
                and ni.shape[0] > ni.shape[1]):
            ok += 1
    assert ok >= 8, f"only {ok}/10 warped+misrotated captures came out clean"


def test_make_stickers_sheet(tmp_path):
    out = build_sticker_sheet(tmp_path / "s.pdf", count=6)
    assert out.read_bytes()[:5] == b"%PDF-"
    png = build_sticker_sheet(tmp_path / "s.png", count=6)
    img = cv2.imread(str(png))
    # the printed stickers are detectable straight off the sheet
    assert len(detect_corner_stickers(img)) >= 4
