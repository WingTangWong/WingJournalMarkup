"""Projection-profile line / word segmentation.

Tesseract does its own layout analysis, but we sometimes want to control the
regions ourselves (metadata cells, a single node's body) - and this is testable
without the binary.
"""

from __future__ import annotations

import cv2
import numpy as np


def _binary(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11
    )


def _runs(mask: np.ndarray, min_gap: int, min_len: int) -> list[tuple[int, int]]:
    on = mask > 0
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > min_gap:
                if i - gap - start >= min_len:
                    runs.append((start, i - gap))
                start = None
    if start is not None and len(on) - start >= min_len:
        runs.append((start, len(on)))
    return runs


def segment_lines(image: np.ndarray, min_line_h: int = 6) -> list[list[int]]:
    """Text-line bounding boxes ``[x, y, w, h]``, top to bottom."""

    binary = _binary(image)
    h, w = binary.shape
    row_ink = (binary > 0).sum(axis=1)
    bands = _runs((row_ink > max(1, 0.01 * w)).astype(np.uint8), min_gap=max(2, h // 60),
                  min_len=min_line_h)
    out: list[list[int]] = []
    for y0, y1 in bands:
        cols = (binary[y0:y1] > 0).sum(axis=0)
        xs = np.where(cols > 0)[0]
        if xs.size == 0:
            continue
        out.append([int(xs[0]), int(y0), int(xs[-1] - xs[0] + 1), int(y1 - y0)])
    return out


def segment_words(image: np.ndarray, line_bbox: list[int]) -> list[list[int]]:
    """Word boxes within one line box (both in the same coords)."""

    x, y, w, h = line_bbox
    binary = _binary(image)[y : y + h, x : x + w]
    col_ink = (binary > 0).sum(axis=0)
    runs = _runs((col_ink > 0).astype(np.uint8), min_gap=max(3, h // 3), min_len=2)
    return [[x + a, y, b - a, h] for a, b in runs]
