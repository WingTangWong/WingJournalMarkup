"""The blank WJM writing sheet: corner markers + empty metadata block."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wingjournal.templates._render import load_font, save_pdf, to_pil
from wingjournal.templates.geometry import (
    _DEFAULT_MARKER_MM,
    METADATA_ROW1_FIELDS,
    METADATA_ROW2_FIELDS,
    PRINT_MARGIN_MM,
    SheetLayout,
    compute_layout,
)
from wingjournal.vision.aruco import DEFAULT_DICT, METADATA_FIELD_IDS, generate_marker

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_INK = (0, 0, 0)
_CAPTION = (150, 150, 150)
_RULE = (208, 208, 208)
_FOOT = (140, 140, 140)
_HINT = (200, 200, 200)


def _dotted_rect(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
    dash: int, gap: int, fill: tuple[int, int, int], width: int = 1,
) -> None:
    """A dashed rectangle (PIL has no dash support), drawn edge by edge."""

    x0, y0, x1, y1 = box
    step = dash + gap
    # top & bottom
    for x in range(x0, x1, step):
        xe = min(x + dash, x1)
        draw.line([x, y0, xe, y0], fill=fill, width=width)
        draw.line([x, y1, xe, y1], fill=fill, width=width)
    # left & right
    for y in range(y0, y1, step):
        ye = min(y + dash, y1)
        draw.line([x0, y, x0, ye], fill=fill, width=width)
        draw.line([x1, y, x1, ye], fill=fill, width=width)


def _draw_registration_mark(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    """A 3-ring concentric square (dark, bright, dark) on a white moat, so the
    grid rules never touch it and it reads as a clean nested contour (spec §11)."""

    for frac, fill in ((1.45, (255, 255, 255)), (1.0, _INK), (0.5, (255, 255, 255)), (0.22, _INK)):
        r = max(1, round(size * frac / 2))
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)


def _draw_metadata_block(
    img: Image.Image, draw: ImageDraw.ImageDraw, layout: SheetLayout, dict_name: str
) -> None:
    """Each metadata field is an ArUco anchor (ids 20-26, one per field) with a
    writing box to its right; the anchor's id names the field and its printed
    size scales the box, so the reader never has to guess field boundaries from
    thin rules (spec §11.3)."""

    x, y, w, h = layout.metadata_rect
    row_h = layout.metadata_row1_h
    box_lw = max(2, layout.dpi // 120)
    cap_font = load_font(max(9, layout.dpi // 20))

    # faint outer frame + row divider for the eye; detection uses the marks
    draw.rectangle([x, y, x + w, y + h], outline=_RULE, width=box_lw)
    draw.line([x, y + row_h, x + w, y + row_h], fill=_RULE, width=box_lw)

    rows = (
        (METADATA_ROW1_FIELDS, layout.row_captions[0]),
        (METADATA_ROW2_FIELDS, layout.row_captions[1]),
    )
    a = layout.field_anchor_px
    for fields, captions in rows:
        for name, label in zip(fields, captions, strict=True):
            ax, ay = layout.field_anchor_xy[name]
            marker = generate_marker(METADATA_FIELD_IDS[name], a, dict_name)
            img.paste(Image.fromarray(marker).convert("RGB"), (ax, ay))

            bx, by, bw, bh = layout.field_box[name]
            # a light box: a writing guide for the eye that Otsu / Tesseract
            # threshold away, so only the ink written inside reaches OCR
            draw.rectangle([bx, by, bx + bw, by + bh], outline=_RULE, width=box_lw)
            draw.text((bx + box_lw + 2, max(y + 1, by - layout.dpi // 18)),
                      label, font=cap_font, fill=_CAPTION)

    if layout.registration_px:
        for cx, cy in layout.registration_xy:
            _draw_registration_mark(draw, cx, cy, layout.registration_px)


def _draw_rules(draw: ImageDraw.ImageDraw, layout: SheetLayout, step_mm: float = 9.0) -> None:
    from wingjournal.templates.geometry import mm_to_px

    step = mm_to_px(step_mm, layout.dpi)
    top = layout.body_top_px
    bottom = layout.marker_xy["BOTTOM_LEFT"][1] - layout.quiet_px
    x0 = layout.margin_px
    x1 = layout.width_px - layout.margin_px
    y = top
    while y < bottom:
        draw.line([x0, y, x1, y], fill=_RULE, width=1)
        y += step


def render_writing_sheet(
    layout: SheetLayout | None = None,
    dict_name: str = DEFAULT_DICT,
    ruled: bool = False,
    page_label: str | None = None,
) -> np.ndarray:
    """Render one writing sheet as a BGR ``np.ndarray`` (what the printer + the
    detector both see)."""

    layout = layout or compute_layout()
    img = Image.new("RGB", (layout.width_px, layout.height_px), "white")
    draw = ImageDraw.Draw(img)

    if ruled:
        _draw_rules(draw, layout)

    # faint dashed frame just outside the corner markers: everything inside is
    # what a capture records, everything outside is not
    from wingjournal.templates.geometry import mm_to_px

    m = layout.margin_px
    d = mm_to_px(1.5, layout.dpi)
    _dotted_rect(
        draw,
        (m - d, m - d, layout.width_px - m + d, layout.height_px - m + d),
        dash=mm_to_px(2.2, layout.dpi), gap=mm_to_px(2.0, layout.dpi),
        fill=_HINT, width=max(1, layout.dpi // 300),
    )

    for role, mid in layout.marker_ids.items():
        marker = generate_marker(mid, layout.marker_px, dict_name)
        img.paste(Image.fromarray(marker).convert("RGB"), layout.marker_xy[role])

    _draw_metadata_block(img, draw, layout, dict_name)

    if page_label:
        def _centred_text(text: str, font, y_centre: int) -> None:
            tb = draw.textbbox((0, 0), text, font=font)
            tx = (layout.width_px - (tb[2] - tb[0])) // 2
            draw.text((tx, y_centre - (tb[3] - tb[1]) // 2 - tb[1]),
                      text, font=font, fill=_FOOT)

        top_band = layout.margin_px + layout.marker_px
        _centred_text(
            page_label, load_font(max(11, layout.dpi // 22)),
            (top_band + layout.body_top_px) // 2,
        )

    return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR


def build_writing_sheet(
    out: str | Path,
    paper: str = "letter",
    pages: int = 1,
    dpi: int = 300,
    dict_name: str = DEFAULT_DICT,
    marker_mm: float = _DEFAULT_MARKER_MM,
    margin_mm: float = PRINT_MARGIN_MM,
    ruled: bool = False,
) -> Path:
    """Write a writing sheet. ``.pdf`` -> PDF (multi-page ok); an image suffix
    (``.png`` etc.) -> a single-page raster."""

    if pages < 1:
        raise ValueError("pages must be >= 1")
    out = Path(out)
    layout = compute_layout(
        paper=paper, dpi=dpi, margin_mm=margin_mm, marker_mm=marker_mm
    )

    if out.suffix.lower() in _RASTER_SUFFIXES:
        if pages != 1:
            raise ValueError("raster output is single-page; use a .pdf for multiple sheets")
        out.parent.mkdir(parents=True, exist_ok=True)
        to_pil(render_writing_sheet(layout, dict_name=dict_name, ruled=ruled)).save(out)
        return out

    sheets = [
        render_writing_sheet(
            layout, dict_name=dict_name, ruled=ruled,
            page_label=(f"sheet {i + 1} / {pages}" if pages > 1 else None),
        )
        for i in range(pages)
    ]
    return save_pdf(sheets, out, dpi)  # save_pdf treats ndarrays as BGR


# backwards-compatible alias
build_writing_sheet_pdf = build_writing_sheet
