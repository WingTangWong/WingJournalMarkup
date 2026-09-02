"""Perspective normalization (spec section 35).

Given an ordered page quadrilateral, compute the homography and warp the page
into normalized coordinates. Output size is derived from the observed quad edge
lengths so we do not stretch the page. An optional upright rotation is folded
into the returned homography, so it always maps raw-image coords onto the image
this function returns.
"""

from __future__ import annotations

import cv2
import numpy as np


def _edge(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def output_size(quad: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = np.asarray(quad, dtype=np.float32)
    width = max(_edge(tl, tr), _edge(bl, br))
    height = max(_edge(tl, bl), _edge(tr, br))
    return max(1, round(width)), max(1, round(height))


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
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normalized_image, homography_3x3)``.

    ``homography`` maps raw-image coordinates onto ``normalized_image``, with any
    ``rotate_degrees`` already composed in.
    """

    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w, h = size if size is not None else output_size(quad)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(quad, dst)

    rot, (w, h) = _rotation_matrix(rotate_degrees, w, h)
    homography = rot @ homography

    warped = cv2.warpPerspective(image, homography, (w, h), flags=cv2.INTER_CUBIC)
    return warped, homography
