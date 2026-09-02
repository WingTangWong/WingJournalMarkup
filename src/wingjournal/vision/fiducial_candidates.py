"""Undecoded square-contour cataloging (spec sections 25-26).

An ArUco payload that will not decode - a blank sticker, a damaged marker, a
hand-drawn scan guide - is still geometric evidence for a page corner. We
catalog square-ish contours without assuming they are fiducials, then let the
hypothesis scorer decide.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.models import DetectedMarker, FiducialCandidate
from wingjournal.vision.boundary import ROLE_SIGNS
from wingjournal.vision.preprocess import Preprocessed


def _square_score(cnt: np.ndarray) -> float:
    """1.0 for a perfect axis-of-symmetry square, decaying with skew/elongation."""

    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return 0.0
    pts = approx.reshape(4, 2).astype(np.float32)
    sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    if sides.min() <= 1e-3:
        return 0.0
    ratio = sides.min() / sides.max()  # 1.0 = equal sides
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(approx)
    fill = cv2.contourArea(approx) / (rw * rh + 1e-6)
    return float(max(0.0, ratio) * max(0.0, min(1.0, fill)))


def find_square_candidates(
    pre: Preprocessed,
    exclude: list[DetectedMarker] | None = None,
    min_area_frac: float = 0.0004,
    max_area_frac: float = 0.05,
    min_square_score: float = 0.6,
) -> list[FiducialCandidate]:
    h, w = pre.gray.shape[:2]
    lo, hi = min_area_frac * h * w, max_area_frac * h * w
    exclude_centers = np.array([m.center for m in (exclude or [])], dtype=np.float32)

    # Close small gaps so a thin, broken square outline reads as one contour.
    ksize = max(3, (min(h, w) // 200) | 1)
    closed = cv2.morphologyEx(
        pre.binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw: list[tuple[np.ndarray, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (lo <= area <= hi):
            continue
        sc = _square_score(cnt)
        if sc < min_square_score:
            continue
        raw.append((cnt, sc))

    if not raw:
        return []

    # Role assignment is relative to the centroid of all square candidates.
    centers = np.array(
        [cv2.minAreaRect(c)[0] for c, _ in raw], dtype=np.float32
    )
    centroid = centers.mean(axis=0)

    out: list[FiducialCandidate] = []
    kept_centers: list[np.ndarray] = []
    for (cnt, sc), center in zip(raw, centers, strict=True):
        if exclude_centers.size:
            d = np.linalg.norm(exclude_centers - center, axis=1).min()
            side = np.sqrt(cv2.contourArea(cnt))
            if d < side:  # overlaps a decoded marker
                continue
        # dedupe near-duplicate contours (inner/outer edge of the same square)
        if kept_centers:
            side = np.sqrt(cv2.contourArea(cnt))
            nearest = min(float(np.linalg.norm(c - center)) for c in kept_centers)
            if nearest < 0.5 * side:
                continue
        kept_centers.append(center)
        rel = center - centroid
        role = max(
            ROLE_SIGNS,
            key=lambda r: float(np.dot(rel, np.array(ROLE_SIGNS[r], np.float32))),
        )
        x, y, bw, bh = cv2.boundingRect(cnt)
        out.append(
            FiducialCandidate(
                bbox=[float(x), float(y), float(bw), float(bh)],
                center=[float(center[0]), float(center[1])],
                decoded=False,
                marker_id=None,
                inferred_role=role,
                reason="square_contour",
                confidence=round(0.35 + 0.4 * sc, 3),
            )
        )
    return out


def refine_squares(
    pre: Preprocessed,
    provisional_quad: list[list[float]] | np.ndarray,
    exclude: list[DetectedMarker] | None = None,
    near_frac: float = 0.18,
) -> list[FiducialCandidate]:
    """Re-detect corner squares near a provisional page frame (spec section 33).

    Uses a relaxed square-score threshold, then keeps only candidates that sit
    close to one of the provisional corners - a boundary estimate lets us trust
    weaker square evidence.
    """

    quad = np.asarray(provisional_quad, dtype=np.float32).reshape(4, 2)
    diag = float(np.linalg.norm(quad[0] - quad[2]))
    if diag <= 0:
        return []
    cands = find_square_candidates(pre, exclude=exclude, min_square_score=0.45)
    kept: list[FiducialCandidate] = []
    for c in cands:
        centre = np.asarray(c.center, dtype=np.float32)
        if min(float(np.linalg.norm(centre - corner)) for corner in quad) < near_frac * diag:
            c.reason = "square_contour_refined"
            kept.append(c)
    return kept
