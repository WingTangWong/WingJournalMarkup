"""Adhesive corner stickers (spec §6, §11.2).

Each sticker is an identical ArUco (``CORNER_STICKER_ID``) with an L-bracket and
a wedge pointing at the page corner. From 3-4 of them we get:

* a page quadrilateral — the wedge tips are the page corners when the user tucked
  them in, otherwise a short extrapolation past each marker;
* a **page-size estimate** — the ArUco is a known physical size, so its scale in
  pixels gives px/mm, and the sticker constellation gives the page in mm.

The stickers are ordinary ArUco to ``detect_markers``; role comes from geometry
and the wedge direction, not the (shared) id.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.models import CornerSticker, DetectedMarker, PageSizeEstimate
from wingjournal.templates.geometry import PAPERS_MM
from wingjournal.vision.aruco import (
    CORNER_STICKER_ARUCO_MM,
    CORNER_STICKER_ID,
    DEFAULT_DICT,
    detect_markers,
)
from wingjournal.vision.boundary import complete_quad_from_three, order_points
from wingjournal.vision.preprocess import to_gray

_ROLE_BY_QUADRANT = {  # sign of the outward vector (x, y-down)
    (-1, -1): "TOP_LEFT",
    (1, -1): "TOP_RIGHT",
    (1, 1): "BOTTOM_RIGHT",
    (-1, 1): "BOTTOM_LEFT",
}


def _marker_side(m: DetectedMarker) -> float:
    c = np.asarray(m.corners, dtype=np.float32)
    return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def _find_bracket(
    gray: np.ndarray, marker: DetectedMarker, outward: np.ndarray
) -> tuple[np.ndarray, bool]:
    """(corner_point, bracket_found).

    Best case: the wedge tip — the ink furthest along ``outward`` just outside
    the marker — is the page corner. When the sticker was stuck on rotated the
    wrong way its wedge no longer points at the corner, so fall back to the
    marker's own outermost corner (a real detected point, still near the page
    corner and rotation-independent — spec §7)."""

    h, w = gray.shape[:2]
    center = np.asarray(marker.center, dtype=np.float32)
    corners = np.asarray(marker.corners, dtype=np.float32)
    side = _marker_side(marker)
    aruco_outer = corners[int(np.argmax((corners - center) @ outward))]
    reach = float((aruco_outer - center) @ outward)
    # geometric guess: past the ArUco outer corner by the known sticker layout
    # (wedge vertex ~ 0.7 marker-widths further out)
    estimate = aruco_outer + outward * (0.7 * side)

    probe = center + outward * side
    r = int(0.9 * side)
    x0, x1 = max(0, int(probe[0] - r)), min(w, int(probe[0] + r))
    y0, y1 = max(0, int(probe[1] - r)), min(h, int(probe[1] + r))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return estimate, False

    region = gray[y0:y1, x0:x1]
    _, ink = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.fillConvexPoly(ink, (corners - [x0, y0]).astype(np.int32), 0)

    ys, xs = np.nonzero(ink)
    if xs.size < max(20, 0.02 * ink.size):
        return estimate, False

    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    proj = (pts - center) @ outward
    tip = pts[int(np.argmax(proj))]
    # a real wedge reaches clearly past the ArUco's own outer corner; a stray
    # blob from a wrong-way-rotated sticker does not (spec §7)
    if float(proj.max()) < 1.1 * reach:
        return estimate, False
    return tip, True


def detect_corner_stickers(
    image: np.ndarray, dict_name: str = DEFAULT_DICT
) -> list[CornerSticker]:
    gray = to_gray(image)
    stickers = [m for m in detect_markers(gray, dict_name) if m.marker_id == CORNER_STICKER_ID]
    if not stickers:
        return []

    centers = np.array([m.center for m in stickers], dtype=np.float32)
    centroid = centers.mean(axis=0)

    out: list[CornerSticker] = []
    for m in stickers:
        c = np.asarray(m.center, dtype=np.float32)
        v = c - centroid
        if np.linalg.norm(v) < 1e-3:
            # single sticker: point from centre toward its own far corner
            corners = np.asarray(m.corners, dtype=np.float32)
            v = corners[int(np.argmax(np.linalg.norm(corners - c, axis=1)))] - c
        outward = v / (np.linalg.norm(v) + 1e-9)

        corner_point, found = _find_bracket(gray, m, outward)
        role = _ROLE_BY_QUADRANT.get(
            (1 if outward[0] >= 0 else -1, 1 if outward[1] >= 0 else -1)
        )
        out.append(CornerSticker(
            marker=m,
            outward=[float(outward[0]), float(outward[1])],
            corner_point=[float(corner_point[0]), float(corner_point[1])],
            bracket_found=found,
            inferred_role=role,
        ))
    return out


def sticker_quad(stickers: list[CornerSticker]) -> np.ndarray | None:
    """Ordered TL, TR, BR, BL from 3-4 stickers' corner points."""

    if len(stickers) >= 4:
        pts = np.array([s.corner_point for s in stickers[:4]], dtype=np.float32)
        return order_points(pts)
    if len(stickers) == 3:
        by_role = {s.inferred_role: np.asarray(s.corner_point, np.float32)
                   for s in stickers if s.inferred_role}
        if len(by_role) == 3:
            return complete_quad_from_three(by_role)
    return None


def estimate_page_size(
    stickers: list[CornerSticker], dict_name: str = DEFAULT_DICT
) -> PageSizeEstimate | None:
    """Physical page size from the sticker ArUco scale + constellation span.

    Each edge is converted with the *local* px/mm of the two stickers it joins,
    which corrects for perspective foreshortening better than one global scale.
    """

    if len(stickers) < 3:
        return None
    quad = sticker_quad(stickers)
    if quad is None:
        return None

    scales = np.array([_marker_side(s.marker) / CORNER_STICKER_ARUCO_MM for s in stickers])
    if np.any(scales < 1e-6):
        return None

    if len(stickers) >= 4:
        # match each ordered quad corner to its sticker's *local* px/mm — corrects
        # perspective foreshortening better than one global scale
        corners = np.array([s.corner_point for s in stickers], dtype=np.float32)
        ppm = scales[[int(np.argmin(np.linalg.norm(corners - q, axis=1))) for q in quad]]
    else:
        ppm = np.full(4, float(np.mean(scales)))
    px_per_mm = float(np.mean(ppm))

    tl, tr, br, bl = quad
    w_mm = float(np.mean([
        np.linalg.norm(tr - tl) / np.mean(ppm[[0, 1]]),
        np.linalg.norm(br - bl) / np.mean(ppm[[3, 2]]),
    ]))
    h_mm = float(np.mean([
        np.linalg.norm(bl - tl) / np.mean(ppm[[0, 3]]),
        np.linalg.norm(br - tr) / np.mean(ppm[[1, 2]]),
    ]))

    best, best_err = None, 1e9
    for name, (pw, ph) in PAPERS_MM.items():
        for a, b in ((pw, ph), (ph, pw)):
            err = abs(w_mm - a) + abs(h_mm - b)
            if err < best_err:
                best, best_err = name, err

    return PageSizeEstimate(
        width_mm=round(w_mm, 1),
        height_mm=round(h_mm, 1),
        px_per_mm=round(px_per_mm, 3),
        method="corner_stickers",
        best_match=best if best_err < 30.0 else None,
        match_error_mm=round(best_err, 1),
    )


def sticker_hypothesis_polygon(
    image: np.ndarray, dict_name: str = DEFAULT_DICT
) -> tuple[np.ndarray | None, list[CornerSticker]]:
    """Convenience for the boundary stage: (quad or None, stickers)."""

    stickers = detect_corner_stickers(image, dict_name)
    return sticker_quad(stickers), stickers
