"""Orientation resolution (spec section 32).

Descending confidence:

  Tier A/B  decoded marker IDs in the constellation  -> exact 0/90/180/270
  Tier F    dominant text-line direction              -> axis only, flip_ambiguous
  (Tier E, metadata-block-as-TOP, needs metadata-block detection - M3.)

``degrees`` is the clockwise rotation to apply to the image to make the page
upright. When ``flip_ambiguous`` is set, the axis is known but 180 could not be
ruled out, so callers should treat ``degrees`` as a best guess.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.models import DetectedMarker, Orientation
from wingjournal.vision.aruco import MARKER_ROLE_IDS
from wingjournal.vision.boundary import ROLES, assign_roles_greedy
from wingjournal.vision.preprocess import Preprocessed

# printed IDs at geometric [TL, TR, BR, BL] on an upright page
_UPRIGHT_ORDER = [MARKER_ROLE_IDS[r] for r in ROLES]


def _from_marker_ids(markers: list[DetectedMarker]) -> Orientation | None:
    """Rotation from the decoded IDs in the constellation.

    Works with as few as 2 markers, as long as exactly one 90-degree rotation is
    consistent with every observed (role -> id) pair.
    """

    roles = assign_roles_greedy(markers)
    if len(roles) < 2:
        return None
    observed = {r: roles[r].marker_id for r in roles}
    if not set(observed.values()) <= set(_UPRIGHT_ORDER):
        return None  # not the canonical ID set
    role_index = {r: i for i, r in enumerate(ROLES)}
    matches = [
        k
        for k in range(4)
        if all(
            np.roll(_UPRIGHT_ORDER, -k)[role_index[r]] == observed[r] for r in observed
        )
    ]
    if len(matches) != 1:
        return None  # ambiguous
    k = matches[0]
    return Orientation(
        degrees=int((k * 90) % 360),
        method="aruco_ids",
        confidence=0.97 if len(roles) == 4 else 0.85,
        flip_ambiguous=False,
    )


def _text_baseline(pre: Preprocessed, polygon: np.ndarray | None) -> Orientation:
    binary = pre.binary
    if polygon is not None:
        mask = np.zeros(binary.shape, np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        binary = cv2.bitwise_and(binary, mask)

    rows = binary.sum(axis=1).astype(np.float64)
    cols = binary.sum(axis=0).astype(np.float64)

    def peakiness(v: np.ndarray) -> float:
        v = v[v > 0]
        if v.size < 4 or v.mean() == 0:
            return 0.0
        return float(v.std() / v.mean())

    # Text-line direction fixes the axis but not which end is the top:
    # 0 vs 180 (and 90 vs 270) are indistinguishable here.
    r, c = peakiness(rows), peakiness(cols)
    if max(r, c) < 1e-6:
        return Orientation(degrees=0, method="assumed", confidence=0.1, flip_ambiguous=True)
    axis_degrees = 0 if r >= c else 90
    return Orientation(
        degrees=axis_degrees, method="text_baseline", confidence=0.45, flip_ambiguous=True
    )


def _from_metadata_block(rectified: np.ndarray) -> Orientation | None:
    """Tier E (spec §32): the segmented metadata block sits at the top of the
    page. Try all four rotations of the provisionally-rectified image and pick
    the one where a metadata block scores best near the top."""

    from wingjournal.recognition.metadata_block import detect_metadata_block
    from wingjournal.vision.aruco import detect_markers

    best: tuple[int, float] | None = None
    for deg in (0, 90, 180, 270):
        img = rotate_upright(rectified, deg)
        mb = detect_metadata_block(img, markers=detect_markers(img))
        if mb is None:
            continue
        near_top = (mb.bbox[1] + mb.bbox[3]) < img.shape[0] * 0.5
        score = mb.confidence * (1.0 if near_top else 0.25)
        if best is None or score > best[1]:
            best = (deg, score)
    if best is None or best[1] < 0.5:
        return None
    return Orientation(
        degrees=best[0], method="metadata_block",
        confidence=round(min(0.95, best[1]), 3), flip_ambiguous=False,
    )


def resolve_orientation(
    pre: Preprocessed,
    markers: list[DetectedMarker],
    boundary_polygon: list[list[float]] | np.ndarray | None = None,
    rectified: np.ndarray | None = None,
) -> Orientation:
    by_ids = _from_marker_ids(markers)
    if by_ids is not None:
        return by_ids

    if rectified is not None:
        by_block = _from_metadata_block(rectified)
        if by_block is not None:
            return by_block

    poly = None
    if boundary_polygon is not None:
        poly = np.asarray(boundary_polygon, dtype=np.float32)
    return _text_baseline(pre, poly)


def rotate_upright(image: np.ndarray, degrees: int) -> np.ndarray:
    deg = degrees % 360
    if deg == 0:
        return image
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"degrees must be a multiple of 90, got {degrees}")
