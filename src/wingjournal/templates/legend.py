"""A one-page printable legend / cheat-sheet for WJM hand markup.

Content is distilled from docs/SPEC-v0-draft.md sections 10-22.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wingjournal.templates._render import load_font, save_pdf
from wingjournal.templates.geometry import PAPERS_MM, mm_to_px

_INK = (20, 20, 20)
_MUTED = (110, 110, 110)
_ACCENT = (40, 70, 140)

SECTIONS: list[tuple[str, list[str]]] = [
    ("Page metadata block (top of page)", [
        "A 2-row segmented box. Blank cells are fine.",
        "Row 1:  DOCUMENT ID | PAGE ID | TOPIC TAGS",
        "Row 2:  LEFT | ABOVE | BELOW | RIGHT   (neighbouring page IDs)",
    ]),
    ("Tag syntax", [
        "#term                 one word",
        "#[term with spaces]    brackets quote spaces",
        "Used for page IDs, document IDs, topics, anchors.",
    ]),
    ("Bullet states", [
        "*  open task        x  completed",
        ">  migrated         <  scheduled",
        "-  note             o  event",
        "!  important        ?  question / research",
        "A later capture that changes the mark updates the",
        "same task (open -> completed), not a new one.",
    ]),
    ("Boxes = nodes", [
        "A plain rectangle is a semantic node.",
        "Add a horizontal divider for a title / body split.",
        "Tags written inside a box belong to that node.",
    ]),
    ("Literal / image region", [
        "Fill all four corners with solid diagonal triangles.",
        "The interior is stored as an image and never parsed",
        "(no OCR, no tags, no arrows).",
    ]),
    ("Diagram arrows (between boxes)", [
        "------      undirected connection",
        "----->      directed        <----->  bidirectional",
        "- - - -     weak / optional relation",
        "Write a label next to the line for an edge label.",
    ]),
    ("Temporal tags", [
        "[DUE: 2026-09-14]",
        "[EVENT: 2026-09-18 14:00]",
        "[RANGE: 2026-09-12 -> 2026-09-19]",
    ]),
    ("Contact box", [
        "+ CONTACT ------------------+",
        "| Jane Smith                |",
        "| jane@example.com          |",
        "| 555-123-4567   Acme Corp  |",
        "+---------------------------+",
    ]),
    ("Anchors & references", [
        "Anchor:     write #ANCHOR on the object",
        "Reference:  -> [#ANCHOR]",
        "Cross-page: document : page : anchor",
        "            e.g. Research:P017:AUTH",
    ]),
    ("Fiducials", [
        "Print 4 ArUco markers (DICT_4X4_50, IDs 0/1/2/3)",
        "near the corners: 0=TL 1=TR 2=BR 3=BL.",
        "Rough placement is fine - the constellation matters,",
        "not each sticker's rotation. Fewer / no markers still",
        "ingest, at lower confidence.",
    ]),
]


def _draw_literal_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=_INK, width=2)
    t = (x1 - x0) // 6
    for cx, cy, dx, dy in (
        (x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1),
    ):
        draw.polygon(
            [(cx, cy), (cx + dx * t, cy), (cx, cy + dy * t)], fill=_INK
        )


def render_legend_image(paper: str = "letter", dpi: int = 200) -> np.ndarray:
    if paper not in PAPERS_MM:
        raise ValueError(f"unknown paper {paper!r}; try {sorted(PAPERS_MM)}")
    w_mm, h_mm = PAPERS_MM[paper]
    W, H = mm_to_px(w_mm, dpi), mm_to_px(h_mm, dpi)
    margin = mm_to_px(14, dpi)

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    h1 = load_font(round(dpi * 0.19), bold=True)
    h2 = load_font(round(dpi * 0.11), bold=True)
    body = load_font(round(dpi * 0.083))
    mono_note = load_font(round(dpi * 0.058))

    draw.text((margin, margin), "Wing Journal Markup - Page Legend", font=h1, fill=_ACCENT)
    top = margin + round(dpi * 0.42)
    draw.line([margin, top - round(dpi * 0.09), W - margin, top - round(dpi * 0.09)],
              fill=_MUTED, width=2)

    col_gap = mm_to_px(12, dpi)
    col_w = (W - 2 * margin - col_gap) // 2
    col_x = [margin, margin + col_w + col_gap]
    col_y = [top, top]
    line_h = round(dpi * 0.112)
    head_h = round(dpi * 0.19)

    # Balance the two columns by total rendered height.
    heights = [head_h + line_h * len(lines) + round(dpi * 0.06) for _, lines in SECTIONS]
    total = sum(heights)
    col = 0
    acc = 0
    for (title, lines), sec_h in zip(SECTIONS, heights, strict=True):
        if col == 0 and acc >= total / 2:
            col = 1
        x = col_x[col]
        y = col_y[col]
        draw.text((x, y), title, font=h2, fill=_INK)
        y += head_h
        for ln in lines:
            draw.text((x + mm_to_px(2, dpi), y), ln, font=body, fill=_INK)
            y += line_h
        col_y[col] = y + round(dpi * 0.13)
        acc += sec_h

    # Literal-region icon next to that section is hard to place generically;
    # draw a small legend icon bottom-right instead.
    icon = mm_to_px(16, dpi)
    ix1, iy1 = W - margin, H - margin
    _draw_literal_icon(draw, (ix1 - icon, iy1 - icon, ix1, iy1))
    draw.text((ix1 - icon - mm_to_px(46, dpi), iy1 - icon // 2),
              "literal image region", font=mono_note, fill=_MUTED)

    draw.text((margin, H - margin + round(dpi * 0.02)),
              "Generated by `wingjournal make-legend` - see docs/SPEC-v0-draft.md",
              font=mono_note, fill=_MUTED)
    return np.asarray(img)[:, :, ::-1].copy()


def build_legend_pdf(out: str | Path, paper: str = "letter", dpi: int = 200) -> Path:
    return save_pdf([render_legend_image(paper=paper, dpi=dpi)], out, dpi)
