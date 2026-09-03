"""Segmented metadata-block detection (spec §11), geometry only.

On a normalized upright page, find the title box at the top and its cell grid:
row 1 has three cells (document id | page id | topic tags), row 2 has four
(left | above | below | right).

Two ways in, tried in order:

1. **registration marks** — the four concentric-square marks at the block
   corners (`vision/registration.py`). Solid ink, so they survive dim or soft
   photos that erase the thin rules. This is the primary path.
2. **ruled lines** — the morphology / projection approach, for sheets printed
   before the marks or hand-drawn blocks.

Text inside the cells is read later (M4). This also gives orientation resolution
a strong "this edge is the top" signal (tier E).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from wingjournal.vision.registration import detect_registration_marks, marks_to_quad


@dataclass
class MetadataBlock:
    bbox: list[float]  # [x, y, w, h] in normalized-page coords
    row_divider_y: float
    row1_cells: list[list[float]] = field(default_factory=list)  # each [x, y, w, h]
    row2_cells: list[list[float]] = field(default_factory=list)
    confidence: float = 0.0
    detection: str = "ruled_lines"  # "registration_marks" | "ruled_lines"
    registration_marks: list[list[float]] = field(default_factory=list)  # [x, y, size, acutance]


def _line_mask(binary: np.ndarray, horizontal: bool, scale: int = 20) -> np.ndarray:
    h, w = binary.shape
    length = max(8, (w if horizontal else h) // scale)
    ksize = (length, 1) if horizontal else (1, length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    eroded = cv2.erode(binary, kernel)
    return cv2.dilate(eroded, kernel)


def _positions(profile: np.ndarray, min_frac: float = 0.4) -> list[int]:
    """Centres of the runs where a projection profile is 'on'."""

    on = profile > (profile.max() * min_frac if profile.max() else 1)
    runs: list[int] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start + i - 1) // 2)
            start = None
    if start is not None:
        runs.append((start + len(on) - 1) // 2)
    return runs


def _row_cells(
    vertical: np.ndarray, bx: int, bw: int, y0: int, y1: int, min_cell: float
) -> list[list[float]]:
    """Cell rectangles for one row band, from that band's vertical dividers."""

    col_profile = vertical[y0:y1, bx : bx + bw].sum(axis=0)
    xs = sorted({bx, *(bx + p for p in _positions(col_profile)), bx + bw})
    cells = [
        [float(xs[i]), float(y0), float(xs[i + 1] - xs[i]), float(y1 - y0)]
        for i in range(len(xs) - 1)
    ]
    return [c for c in cells if c[2] >= min_cell]


def _cells_equal(bx: float, by: float, bw: float, bh: float, divider_y: float) -> tuple[list, list]:
    """3 equal cells in row 1, 4 in row 2 — matches the printed sheet."""

    def split(n: int, y0: float, y1: float) -> list[list[float]]:
        return [
            [bx + bw * i / n, y0, bw / n, y1 - y0] for i in range(n)
        ]

    return split(3, by, divider_y), split(4, divider_y, by + bh)


def _from_registration_marks(
    gray: np.ndarray, search_frac: float, exclude: list | None
) -> MetadataBlock | None:
    h, w = gray.shape[:2]
    band = (0, 0, w, max(1, int(h * search_frac)))
    marks = detect_registration_marks(gray, roi=band, exclude=exclude, expected=4)
    quad = marks_to_quad(marks)
    if quad is None:
        return None

    xs = quad[:, 0]
    ys = quad[:, 1]
    bx, by = float(xs.min()), float(ys.min())
    bw, bh = float(xs.max() - bx), float(ys.max() - by)
    if bw < 0.35 * w or bh < 6 or bw / max(bh, 1) < 2.0:
        return None

    divider_y = by + bh / 2.0
    r1, r2 = _cells_equal(bx, by, bw, bh, divider_y)
    acut = float(np.mean([m.acutance for m in marks]))
    return MetadataBlock(
        bbox=[bx, by, bw, bh],
        row_divider_y=divider_y,
        row1_cells=r1,
        row2_cells=r2,
        confidence=round(0.75 + 0.25 * acut, 3),
        detection="registration_marks",
        registration_marks=[
            [round(m.center[0], 1), round(m.center[1], 1), round(m.size, 1), m.acutance]
            for m in marks
        ],
    )


def detect_metadata_block(
    normalized_image: np.ndarray,
    search_frac: float = 0.42,
    marker_boxes: list | None = None,
) -> MetadataBlock | None:
    """``marker_boxes`` = (x, y, w, h) of the corner ArUco markers, so the
    registration-mark search skips them."""

    gray = (
        normalized_image
        if normalized_image.ndim == 2
        else cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    )
    h, w = gray.shape[:2]

    via_marks = _from_registration_marks(gray, search_frac, marker_boxes)
    if via_marks is not None:
        return via_marks

    top = gray[: int(h * search_frac), :]
    binary = cv2.adaptiveThreshold(
        top, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11
    )

    horizontal = _line_mask(binary, horizontal=True)
    vertical = _line_mask(binary, horizontal=False)
    grid = cv2.bitwise_or(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, int, int] | None = None
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.55 * w or bh < 0.02 * h:
            continue
        # the block spans the gap between the top markers with two 1/4" rows
        # (~12:1 at defaults); allow headroom for thinner rows / wider paper
        if not (2.0 <= bw / max(bh, 1) <= 24.0):
            continue
        if best is None or bw * bh > best[2] * best[3]:
            best = (x, y, bw, bh)
    if best is None:
        return None

    bx, by, bw, bh = best
    margin = max(2, bh // 8)
    row_profile = horizontal[by + margin : by + bh - margin, bx : bx + bw].sum(axis=1)
    if row_profile.size == 0 or row_profile.max() == 0:
        divider_y = int(by + bh / 2)
    else:
        divider_y = by + margin + int(np.argmax(row_profile))

    min_cell = 0.04 * bw
    r1 = _row_cells(vertical, bx, bw, by, divider_y, min_cell)
    r2 = _row_cells(vertical, bx, bw, divider_y, by + bh, min_cell)

    # confidence: rewards the expected 3 / 4 cell split and a wide box
    cell_score = 0.5 * (len(r1) == 3) + 0.5 * (len(r2) == 4)
    width_score = min(1.0, bw / (0.85 * w))
    return MetadataBlock(
        bbox=[float(bx), float(by), float(bw), float(bh)],
        row_divider_y=float(divider_y),
        row1_cells=r1,
        row2_cells=r2,
        confidence=round(0.4 + 0.4 * cell_score + 0.2 * width_score, 3),
    )
