"""The blank WJM writing sheet: corner markers + empty metadata block."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wingjournal.templates._render import load_font, save_pdf, to_pil
from wingjournal.templates.geometry import SheetLayout, compute_layout
from wingjournal.vision.aruco import DEFAULT_DICT, generate_marker

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_INK = (0, 0, 0)
_CAPTION = (150, 150, 150)
_RULE = (208, 208, 208)
_FOOT = (140, 140, 140)


def _draw_metadata_block(draw: ImageDraw.ImageDraw, layout: SheetLayout) -> None:
    x, y, w, h = layout.metadata_rect
    row_h = layout.metadata_row1_h
    lw = max(2, layout.dpi // 150)
    cap_font = load_font(max(12, layout.dpi // 13))

    draw.rectangle([x, y, x + w, y + h], outline=_INK, width=lw)
    draw.line([x, y + row_h, x + w, y + row_h], fill=_INK, width=lw)

    for captions, y0 in (
        (layout.row_captions[0], y),
        (layout.row_captions[1], y + row_h),
    ):
        n = len(captions)
        for c in range(1, n):
            cx = x + round(w * c / n)
            draw.line([cx, y0, cx, y0 + row_h], fill=_INK, width=lw)
        pad = layout.dpi // 40
        for c, label in enumerate(captions):
            cx = x + round(w * c / n)
            draw.text((cx + pad, y0 + pad), label, font=cap_font, fill=_CAPTION)


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

    for role, mid in layout.marker_ids.items():
        marker = generate_marker(mid, layout.marker_px, dict_name)
        img.paste(Image.fromarray(marker).convert("RGB"), layout.marker_xy[role])

    _draw_metadata_block(draw, layout)

    foot = (
        f"Wing Journal Markup - writing sheet - {dict_name} - "
        f"marker IDs 0/1/2/3 = TL/TR/BR/BL"
    )
    foot_font = load_font(max(11, layout.dpi // 22))
    fb = draw.textbbox((0, 0), foot, font=foot_font)
    fx = (layout.width_px - (fb[2] - fb[0])) // 2
    fy = layout.height_px - layout.margin_px // 2 - (fb[3] - fb[1]) // 2
    draw.text((fx, fy), foot, font=foot_font, fill=_FOOT)

    if page_label:
        lf = load_font(max(11, layout.dpi // 20))
        lb = draw.textbbox((0, 0), page_label, font=lf)
        lx = (layout.width_px - (lb[2] - lb[0])) // 2
        draw.text((lx, layout.margin_px // 3), page_label, font=lf, fill=_FOOT)

    return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR


def build_writing_sheet(
    out: str | Path,
    paper: str = "letter",
    pages: int = 1,
    dpi: int = 300,
    dict_name: str = DEFAULT_DICT,
    marker_mm: float = 18.0,
    margin_mm: float = 12.0,
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
