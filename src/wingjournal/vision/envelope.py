"""Content-anchored page-frame guesses (spec sections 27-28, orientation tier G).

When there are too few fiducials to pin the page down, the catalogued content
still bounds where the page must be. These hypotheses feed the scorer alongside
the fiducial ones - they are guesses, not decisions.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.models import PageHypothesis
from wingjournal.vision.boundary import order_points
from wingjournal.vision.preprocess import Preprocessed

# long-side / short-side of common page stock
_PAGE_ASPECTS = (279.4 / 215.9, 297.0 / 210.0)


def significant_contours(pre: Preprocessed) -> list[np.ndarray]:
    """Foreground contours big enough to be structure, not speckle or the frame."""

    h, w = pre.gray.shape[:2]
    lo, hi = 0.0002 * h * w, 0.9 * h * w
    contours, _ = cv2.findContours(pre.binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if lo <= cv2.contourArea(c) <= hi]


def content_points(pre: Preprocessed) -> np.ndarray | None:
    cnts = significant_contours(pre)
    if not cnts:
        return None
    return np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)


def _clamp_quad(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    quad = quad.copy()
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad


def envelope_hypotheses(pre: Preprocessed) -> list[PageHypothesis]:
    pts = content_points(pre)
    if pts is None or len(pts) < 8:
        return []
    h, w = pre.gray.shape[:2]
    x0, y0, cw, ch = cv2.boundingRect(pts)
    if cw < 2 or ch < 2:
        return []
    margin = 0.06 * max(cw, ch)
    out: list[PageHypothesis] = []

    # 1. content bbox grown by a symmetric margin
    bx0, by0 = x0 - margin, y0 - margin
    bx1, by1 = x0 + cw + margin, y0 + ch + margin
    out.append(
        PageHypothesis(
            polygon=_clamp_quad(
                np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], np.float32), w, h
            ).tolist(),
            source="content_envelope",
        )
    )

    # 2. min-area (rotated) rect of the content, grown a little
    rect = cv2.minAreaRect(pts)
    (rcx, rcy), (rw, rh), rangle = rect
    grown = ((rcx, rcy), (rw + 2 * margin, rh + 2 * margin), rangle)
    out.append(
        PageHypothesis(
            polygon=_clamp_quad(order_points(cv2.boxPoints(grown)), w, h).tolist(),
            source="content_envelope_rot",
        )
    )

    # 3. WJM content clusters near the top; anchor there and extend to a
    #    plausible page aspect (both stock ratios).
    page_w = cw + 2 * margin
    for aspect in _PAGE_ASPECTS:
        page_h = page_w * aspect
        ax0, ay0 = x0 - margin, y0 - margin
        quad = np.array(
            [[ax0, ay0], [ax0 + page_w, ay0], [ax0 + page_w, ay0 + page_h], [ax0, ay0 + page_h]],
            np.float32,
        )
        out.append(
            PageHypothesis(
                polygon=_clamp_quad(quad, w, h).tolist(),
                source="content_aspect",
            )
        )
    return out
