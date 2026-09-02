"""Image preprocessing (spec section 48: PREPROCESS).

Grayscale, contrast normalization, and binary / edge / contour extraction.
Kept deliberately simple and parameterized; tuning is a later roadmap item.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Preprocessed:
    gray: np.ndarray
    normalized: np.ndarray  # CLAHE-equalized grayscale
    binary: np.ndarray  # inverted adaptive threshold (ink = 255)
    edges: np.ndarray


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def binarize(gray: np.ndarray, block_size: int = 35, c: int = 11) -> np.ndarray:
    block_size = max(3, block_size | 1)  # must be odd and >= 3
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def preprocess(image: np.ndarray) -> Preprocessed:
    gray = to_gray(image)
    normalized = normalize_contrast(gray)
    return Preprocessed(
        gray=gray,
        normalized=normalized,
        binary=binarize(normalized),
        edges=edges(normalized),
    )


def find_quads(binary_or_edges: np.ndarray, min_area_frac: float = 0.001) -> list[np.ndarray]:
    """Return 4-point convex contours, largest first, as float32 (4, 2) arrays."""

    h, w = binary_or_edges.shape[:2]
    min_area = min_area_frac * h * w
    contours, _ = cv2.findContours(
        binary_or_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    quads: list[tuple[float, np.ndarray]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append((area, approx.reshape(4, 2).astype(np.float32)))
    quads.sort(key=lambda t: t[0], reverse=True)
    return [q for _, q in quads]
