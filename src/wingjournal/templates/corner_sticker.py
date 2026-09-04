"""Printable adhesive corner stickers (spec §6, §11.2).

Every sticker is identical: an ArUco (``CORNER_STICKER_ID``) with an L-bracket
and a wedge on two adjacent edges. Peel one, rotate it so the wedge points into
a paper corner, and stick — four of them turn any sheet into a scannable WJM
page without proprietary paper. The ArUco is a fixed physical size, so the
detector can back out how big the page is.

``make-stickers`` prints a grid of them with cut guides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wingjournal.templates._render import load_font, save_pdf, to_pil
from wingjournal.templates.geometry import PAPERS_MM, mm_to_px
from wingjournal.vision.aruco import (
    CORNER_STICKER_ARUCO_MM,
    CORNER_STICKER_ID,
    DEFAULT_DICT,
    generate_marker,
)

_INK = (0, 0, 0)
_CUT = (170, 170, 170)
_FOOT = (120, 120, 120)

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# sticker geometry, in mm, for the canonical "points to TOP_LEFT" orientation
_WEDGE_MM = 8.0     # solid photo-corner triangle: legs, 90-deg vertex = the corner
_BRACKET_MM = 0.8   # thin alignment rule continuing along the two outer edges
_BRACKET_LEN_MM = 13.0
_ARUCO_ORIGIN_MM = 7.0  # ArUco top-left inset (keeps its quiet zone off the wedge)


def render_corner_sticker(
    sticker_mm: float = 26.0, dpi: int = 300, dict_name: str = DEFAULT_DICT
) -> np.ndarray:
    """One sticker as a BGR ``np.ndarray``. The solid wedge points out through
    the top-left; rotate the physical sticker 90° per corner."""

    S = mm_to_px(sticker_mm, dpi)
    Wg = mm_to_px(_WEDGE_MM, dpi)
    B = max(1, mm_to_px(_BRACKET_MM, dpi))
    Bl = mm_to_px(_BRACKET_LEN_MM, dpi)
    A = mm_to_px(CORNER_STICKER_ARUCO_MM, dpi)
    o = mm_to_px(_ARUCO_ORIGIN_MM, dpi)

    img = Image.new("RGB", (S, S), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, S - 1, S - 1], outline=_CUT, width=max(1, dpi // 300))

    # solid photo-corner wedge — its 90-deg vertex is tucked into the paper corner
    draw.polygon([(0, 0), (Wg, 0), (0, Wg)], fill=_INK)
    # thin rule continuing along the two outer edges, for eyeball alignment
    draw.rectangle([0, 0, Bl, B], fill=_INK)
    draw.rectangle([0, 0, B, Bl], fill=_INK)

    # ArUco toward the bottom-right, quiet zone clear of the wedge and every edge
    marker = generate_marker(CORNER_STICKER_ID, A, dict_name)
    img.paste(Image.fromarray(marker).convert("RGB"), (o, o))

    return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR


def _instructions(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, dpi: int) -> int:
    title_font = load_font(max(16, dpi // 12), bold=True)
    body_font = load_font(max(12, dpi // 18))
    draw.text((x, y), "WJM corner stickers", font=title_font, fill=_INK)
    y += max(20, dpi // 9)
    lines = [
        "Peel one sticker. Rotate it so the solid wedge points diagonally",
        "out through a paper corner, and press it down.",
        "Do the same in all four corners, then photograph or scan the page.",
        f"Every sticker is the same (ArUco id {CORNER_STICKER_ID}); its rotation is the corner.",
    ]
    for ln in lines:
        draw.text((x, y), ln, font=body_font, fill=_FOOT)
        y += max(15, dpi // 15)

    # a tiny 4-corner diagram
    y += dpi // 20
    box = mm_to_px(28, dpi)
    draw.rectangle([x, y, x + box, y + box], outline=_CUT, width=max(1, dpi // 300))
    t = mm_to_px(4, dpi)
    for (cx, cy, dx, dy) in (
        (x, y, 1, 1), (x + box, y, -1, 1), (x + box, y + box, -1, -1), (x, y + box, 1, -1),
    ):
        draw.polygon([(cx, cy), (cx + dx * t, cy), (cx, cy + dy * t)], fill=_INK)
    return y + box + dpi // 12


def build_sticker_sheet(
    out: str | Path,
    paper: str = "letter",
    sticker_mm: float = 26.0,
    count: int = 12,
    dpi: int = 300,
    dict_name: str = DEFAULT_DICT,
    margin_mm: float = 12.0,
    gap_mm: float = 6.0,
) -> Path:
    """A page of identical corner stickers with cut guides."""

    if count < 1:
        raise ValueError("count must be >= 1")
    out = Path(out)
    try:
        pw_mm, ph_mm = PAPERS_MM[paper]
    except KeyError as exc:
        raise ValueError(f"unknown paper {paper!r}") from exc

    W = mm_to_px(pw_mm, dpi)
    H = mm_to_px(ph_mm, dpi)
    m = mm_to_px(margin_mm, dpi)
    gap = mm_to_px(gap_mm, dpi)
    s = mm_to_px(sticker_mm, dpi)

    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    top = _instructions(draw, m, m, W - 2 * m, dpi)

    sticker = to_pil(render_corner_sticker(sticker_mm, dpi, dict_name))
    x = m
    y = top
    placed = 0
    while placed < count and y + s <= H - m:
        page.paste(sticker, (x, y))
        placed += 1
        x += s + gap
        if x + s > W - m:
            x = m
            y += s + gap

    foot = f"WJM corner stickers - ArUco id {CORNER_STICKER_ID} - {sticker_mm:g} mm - {dict_name}"
    ff = load_font(max(11, dpi // 22))
    fb = draw.textbbox((0, 0), foot, font=ff)
    fx = (W - (fb[2] - fb[0])) // 2
    draw.text((fx, H - m // 2 - (fb[3] - fb[1]) // 2), foot, font=ff, fill=_FOOT)

    bgr = np.asarray(page)[:, :, ::-1].copy()
    if out.suffix.lower() in _RASTER_SUFFIXES:
        out.parent.mkdir(parents=True, exist_ok=True)
        to_pil(bgr).save(out)
        return out
    return save_pdf([bgr], out, dpi)
