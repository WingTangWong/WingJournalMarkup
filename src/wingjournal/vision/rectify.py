"""Perspective normalization (spec section 35; see docs/COORDINATES.md).

Given an ordered page quadrilateral, compute the homography and warp the page
into normalized coordinates. The output is scaled to a fixed target size (longer
side = ``target_long_px``) so repeated captures of one page are directly
comparable. An optional upright rotation is folded into the returned homography,
so it always maps raw-image coords onto the image this function returns.
"""

from __future__ import annotations

import cv2
import numpy as np

# default normalized-page long side, in pixels (~145 DPI for US Letter)
TARGET_LONG_PX = 1600

# adaptive normalized long side (spec §9.1): never below MIN, never far past the
# page's real pixel span in the capture (warpPerspective's cubic upscale past
# that just interpolates), capped at MAX so the buffer stays manageable.
MIN_LONG_PX = 1600
MAX_LONG_PX = 2800

# plausible page aspect ratios (long / short): Letter 1.294, A4 1.414
_ASPECT_RANGE = (1.15, 1.6)


def adaptive_long_px(
    quad: np.ndarray,
    lo: int = MIN_LONG_PX,
    hi: int = MAX_LONG_PX,
) -> int:
    """Target long side = the page's own pixel span in the raw capture, clamped
    to ``[lo, hi]``. Upsampling past the captured span only interpolates;
    downsampling far below it throws away readable ink. Mirrors
    ``MobileDeviceDemo/js/worker.js``."""

    width, height = observed_size(quad)
    return int(np.clip(round(max(width, height)), lo, hi))


def _edge(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def observed_size(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = np.asarray(quad, dtype=np.float32)
    width = max(_edge(tl, tr), _edge(bl, br))
    height = max(_edge(tl, bl), _edge(tr, br))
    return width, height


def output_size(quad: np.ndarray, target_long_px: int = TARGET_LONG_PX) -> tuple[int, int]:
    """Fixed (w, h) for the normalized page: longer side == target_long_px,
    aspect ratio taken from the quad and clamped to the plausible page range."""

    width, height = observed_size(quad)
    if min(width, height) < 1e-3:
        return max(1, target_long_px), max(1, target_long_px)
    aspect = max(width, height) / min(width, height)
    aspect = float(np.clip(aspect, *_ASPECT_RANGE))
    long_px = target_long_px
    short_px = max(1, round(long_px / aspect))
    return (short_px, long_px) if height >= width else (long_px, short_px)


def _rotation_matrix(degrees: int, w: int, h: int) -> tuple[np.ndarray, tuple[int, int]]:
    """3x3 matrix + new (w, h) for a clockwise multiple-of-90 image rotation."""

    d = degrees % 360
    if d == 0:
        return np.eye(3, dtype=np.float64), (w, h)
    if d == 90:
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], np.float64), (h, w)
    if d == 180:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64), (w, h)
    if d == 270:
        return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], np.float64), (h, w)
    raise ValueError(f"rotate_degrees must be a multiple of 90, got {degrees}")


def rectify(
    image: np.ndarray,
    quad: np.ndarray,
    size: tuple[int, int] | None = None,
    rotate_degrees: int = 0,
    target_long_px: int = TARGET_LONG_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normalized_image, homography_3x3)``.

    ``homography`` maps raw-image coordinates onto ``normalized_image``, with any
    ``rotate_degrees`` already composed in. ``size`` overrides the fixed target.
    """

    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w, h = size if size is not None else output_size(quad, target_long_px)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(quad, dst)

    rot, (w2, h2) = _rotation_matrix(rotate_degrees, w, h)
    homography = rot @ homography

    # INTER_AREA is the right resampler when we're shrinking the page (it
    # box-filters instead of aliasing); INTER_CUBIC's negative lobes overshoot
    # hard ink edges near 1:1 and Tesseract's threshold then mangles them, so
    # only reach for it on a real upscale.
    src_long = max(observed_size(quad))
    ratio = max(w, h) / src_long if src_long else 1.0
    if ratio < 0.98:
        interp = cv2.INTER_AREA
    elif ratio > 1.25:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR

    warped = cv2.warpPerspective(image, homography, (w2, h2), flags=interp)
    return warped, homography
