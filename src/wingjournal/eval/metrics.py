"""Scoring metrics for the evaluation harness."""

from __future__ import annotations

import cv2
import numpy as np


def polygon_iou(
    a: list[list[float]] | np.ndarray,
    b: list[list[float]] | np.ndarray,
    shape: tuple[int, int],
) -> float:
    """Intersection-over-union of two quads, rasterized at ``shape`` (h, w)."""

    h, w = shape[:2]
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.asarray(a, dtype=np.int32)], 255)
    cv2.fillPoly(mb, [np.asarray(b, dtype=np.int32)], 255)
    inter = int(np.count_nonzero(cv2.bitwise_and(ma, mb)))
    union = int(np.count_nonzero(cv2.bitwise_or(ma, mb)))
    return inter / union if union else 0.0
