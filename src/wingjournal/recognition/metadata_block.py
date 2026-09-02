"""Segmented metadata-block detection (spec §11), geometry only.

On a normalized upright page, find the ruled box at the top and its cell grid:
row 1 has three cells (document id | page id | topic tags), row 2 has four
(left | above | below | right). Text inside the cells is read later (M4). This
also gives orientation resolution a strong "this edge is the top" signal
(tier E).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MetadataBlock:
    bbox: list[float]  # [x, y, w, h] in normalized-page coords
    row_divider_y: float
    row1_cells: list[list[float]] = field(default_factory=list)  # each [x, y, w, h]
    row2_cells: list[list[float]] = field(default_factory=list)
    confidence: float = 0.0


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


def detect_metadata_block(
    normalized_image: np.ndarray, search_frac: float = 0.42
) -> MetadataBlock | None:
    gray = (
        normalized_image
        if normalized_image.ndim == 2
        else cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    )
    h, w = gray.shape[:2]
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
        if not (2.0 <= bw / max(bh, 1) <= 14.0):
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
