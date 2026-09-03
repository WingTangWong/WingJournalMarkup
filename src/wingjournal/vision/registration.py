"""Concentric-square registration marks (spec §11).

A registration mark is a 3-ring bullseye — a solid dark square, a bright square
inside it, and a small dark square at the centre. Four of them at the corners of
the metadata block let the detector lock onto the field grid even when the thin
ruled lines are lost to poor lighting or a soft photo.

The nested-contour signature (dark → bright → dark, all roughly square and
concentric) is cheap to find and, being solid ink rather than hairlines, holds
up where the rules do not. Each mark also doubles as a **sharpness probe**: the
edge between its rings is a known step, so a soft edge there means a soft photo
(see ``wingjournal.vision.sharpness``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wingjournal.vision.boundary import order_points
from wingjournal.vision.preprocess import to_gray


@dataclass
class RegistrationMark:
    center: list[float]        # [x, y] in the searched image's coords
    size: float               # outer side length, px
    acutance: float = 0.0     # 0..1 edge sharpness of the ring transitions
    rings: int = 2            # nested rings actually resolved (2 or 3)


def _is_squareish(
    contour: np.ndarray, tol: float = 0.35
) -> tuple[float, tuple[float, float]] | None:
    """(side, centre) if ``contour`` is roughly a square, else ``None``."""

    area = cv2.contourArea(contour)
    if area < 16:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if min(w, h) == 0 or abs(w - h) > tol * max(w, h):
        return None
    fill = area / (w * h)
    if fill < 0.6:  # a square (filled or ring-outline) fills its bbox well
        return None
    return float(max(w, h)), (x + w / 2.0, y + h / 2.0)


def _edge_acutance(gray: np.ndarray, cx: float, cy: float, size: float) -> float:
    """Sharpness of the ring transitions, 0 (mush) .. 1 (crisp).

    Sample a horizontal and a vertical line through the centre and look at the
    steepest local intensity step: a crisp mark rises ~full contrast in a pixel
    or two, a soft one ramps over many.
    """

    h, w = gray.shape[:2]
    half = max(4, int(size * 0.7))
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half) + 1)
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half) + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 0.0

    profiles = [
        gray[int(np.clip(cy, 0, h - 1)), x0:x1].astype(np.float32),
        gray[y0:y1, int(np.clip(cx, 0, w - 1))].astype(np.float32),
    ]
    best = 0.0
    for p in profiles:
        if p.size < 6:
            continue
        p = cv2.GaussianBlur(p.reshape(1, -1), (1, 3), 0).ravel()
        span = float(p.max() - p.min())
        if span < 30:  # washed out — treat as unsharp
            continue
        grad = np.abs(np.diff(p))
        # steepest step as a fraction of the mark's own contrast
        best = max(best, float(grad.max()) / span)
    return float(np.clip(best, 0.0, 1.0))


def detect_registration_marks(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    expected: int = 4,
    exclude: list[tuple[float, float, float, float]] | None = None,
    min_size_frac: float = 0.006,
    max_size_frac: float = 0.06,
) -> list[RegistrationMark]:
    """Find concentric-square marks. ``roi`` = (x, y, w, h) limits the search;
    ``exclude`` boxes (e.g. the ArUco markers) are ignored."""

    gray = to_gray(image)
    H, W = gray.shape[:2]
    ox, oy = 0, 0
    if roi is not None:
        ox, oy, rw, rh = roi
        gray = gray[oy:oy + rh, ox:ox + rw]
        H, W = gray.shape[:2]
    if H < 8 or W < 8:
        return []

    lo = min_size_frac * max(H, W)
    hi = max_size_frac * max(H, W)

    # a global threshold, not adaptive: the marks are solid ink, and adaptive
    # thresholding hollows out any fill larger than its window (cf. literal_box).
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    excl = exclude or []

    def hidden(cx: float, cy: float) -> bool:
        gx, gy = cx + ox, cy + oy
        return any(bx <= gx <= bx + bw and by <= gy <= by + bh for bx, by, bw, bh in excl)

    out: list[RegistrationMark] = []
    for i, cnt in enumerate(contours):
        # outer ring: an outermost dark blob (no parent)
        if hierarchy[i][3] != -1:
            continue
        outer = _is_squareish(cnt)
        if outer is None:
            continue
        side, (cx, cy) = outer
        if not (lo <= side <= hi) or hidden(cx, cy):
            continue

        # a child (the bright square) roughly concentric, 0.12..0.72 of the side
        near = 0.25 * side
        child = hierarchy[i][2]
        rings = 1
        while child != -1:
            c = _is_squareish(contours[child])
            if c is not None:
                cs, (ccx, ccy) = c
                if (0.12 * side <= cs <= 0.72 * side
                        and abs(ccx - cx) < near and abs(ccy - cy) < near):
                    rings += 1
            child = hierarchy[child][0]
        if rings < 2:
            continue

        out.append(RegistrationMark(
            center=[float(cx + ox), float(cy + oy)],
            size=side,
            acutance=round(_edge_acutance(gray, cx, cy, side), 3),
            rings=min(rings, 3),
        ))

    # de-dupe near-identical detections, keep the crispest
    out.sort(key=lambda m: (-m.rings, -m.acutance, -m.size))
    deduped: list[RegistrationMark] = []
    for m in out:
        dup = any(
            abs(m.center[0] - d.center[0]) < 0.5 * m.size
            and abs(m.center[1] - d.center[1]) < 0.5 * m.size
            for d in deduped
        )
        if not dup:
            deduped.append(m)

    if len(deduped) > expected:
        deduped = deduped[:expected]
    return deduped


def marks_to_quad(marks: list[RegistrationMark]) -> np.ndarray | None:
    """4 marks → an ordered TL, TR, BR, BL quad of their centres, else ``None``."""

    if len(marks) != 4:
        return None
    pts = np.array([m.center for m in marks], dtype=np.float32)
    quad = order_points(pts)
    # sanity: convex, non-degenerate
    if cv2.contourArea(quad) < 1.0:
        return None
    return quad
