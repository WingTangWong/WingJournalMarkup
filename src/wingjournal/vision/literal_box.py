"""Literal / static-image region detection (spec §16, §36).

A rectangle whose four corners carry solid diagonal black fills (like old
photo-album mounts) is a ``LiteralAsset``: its interior is extracted as an image
and must NOT be semantically parsed. Detection + masking runs *before* the
detailed recognition stages.

The corners are the whole signal: a real corner fill is a *solid triangle*, so
a probe placed on the inward diagonal lands in ink, whereas a plain ruled box
(two thin lines meeting in an L) leaves that probe on white paper.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.models import LiteralAsset


def _ink(binary: np.ndarray, cx: int, cy: int, half: int) -> float:
    h, w = binary.shape
    patch = binary[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]
    return float((patch > 0).mean()) if patch.size else 0.0


def _corner_scores(
    binary: np.ndarray, x: int, y: int, bw: int, bh: int, size: int
) -> tuple[float, float]:
    """``(min corner-tip fill, min inward-wedge fill)`` over the four corners."""

    half = max(2, size // 2)
    inward = max(3, int(size * 0.8))
    corners = ((x, y, 1, 1), (x + bw, y, -1, 1), (x + bw, y + bh, -1, -1), (x, y + bh, 1, -1))
    tips, wedges = [], []
    for cx, cy, dx, dy in corners:
        tips.append(_ink(binary, cx + dx * half, cy + dy * half, half))
        wedges.append(_ink(binary, cx + dx * inward, cy + dy * inward, max(2, size // 3)))
    return min(tips), min(wedges)


def detect_literal_assets(
    normalized_image: np.ndarray,
    min_area_frac: float = 0.004,
    corner_frac: float = 0.09,
    min_tip_fill: float = 0.5,
    min_wedge_fill: float = 0.3,
    max_edge_fill: float = 0.3,
) -> list[LiteralAsset]:
    gray = (
        normalized_image
        if normalized_image.ndim == 2
        else cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    )
    h, w = gray.shape[:2]
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11
    )
    # adaptive threshold only sees edges; use a global one to measure solid fills
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[LiteralAsset] = []
    seen: list[tuple[int, int]] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area_frac * h * w or area > 0.9 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 0.08 * w or bh < 0.08 * h:
            continue
        if any(abs(x - sx) < 15 and abs(y - sy) < 15 for sx, sy in seen):
            continue

        size = max(4, int(min(bw, bh) * corner_frac))
        tip, wedge = _corner_scores(binary, x, y, bw, bh, size)
        if tip < min_tip_fill or wedge < min_wedge_fill:
            continue
        edge = max(
            _ink(binary, x + bw // 2, y, size),
            _ink(binary, x + bw // 2, y + bh, size),
            _ink(binary, x, y + bh // 2, size),
            _ink(binary, x + bw, y + bh // 2, size),
        )
        if edge > max_edge_fill:  # a filled shape / marker, not a thin-ruled box
            continue

        seen.append((x, y))
        out.append(LiteralAsset(
            bbox=[float(x), float(y), float(bw), float(bh)],
            confidence=round(min(1.0, 0.5 * tip + 0.5 * wedge) * (1 - edge), 3),
        ))
    return out


def mask_literals(image: np.ndarray, assets: list[LiteralAsset], fill: int = 255) -> np.ndarray:
    """Blank the interior of every literal box so later stages skip it."""

    out = image.copy()
    for a in assets:
        x, y, w, h = (int(round(v)) for v in a.bbox)
        pad = max(2, int(min(w, h) * 0.12))
        colour = (fill, fill, fill) if out.ndim == 3 else fill
        cv2.rectangle(out, (x + pad, y + pad), (x + w - pad, y + h - pad), colour, -1)
    return out


def crop_literal(image: np.ndarray, asset: LiteralAsset) -> np.ndarray:
    x, y, w, h = (int(round(v)) for v in asset.bbox)
    pad = max(2, int(min(w, h) * 0.1))
    hh, ww = image.shape[:2]
    return image[max(0, y + pad) : min(hh, y + h - pad), max(0, x + pad) : min(ww, x + w - pad)]
