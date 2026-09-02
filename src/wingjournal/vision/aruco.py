"""ArUco marker detection and generation (spec sections 5, 24).

Uses the OpenCV >= 4.7 object-oriented ArUco API
(``cv2.aruco.ArucoDetector``).
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.models import DetectedMarker
from wingjournal.vision.preprocess import to_gray

# Default dictionary. 4x4/50 is a good balance of robustness and print size
# for hand-placed corner stickers.
DEFAULT_DICT = "DICT_4X4_50"

# Canonical marker-id -> page-corner-role convention, used by both the printable
# templates and orientation resolution.
MARKER_ROLE_IDS: dict[str, int] = {
    "TOP_LEFT": 0,
    "TOP_RIGHT": 1,
    "BOTTOM_RIGHT": 2,
    "BOTTOM_LEFT": 3,
}

_DICT_NAMES = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def available_dictionaries() -> list[str]:
    return sorted(_DICT_NAMES)


def get_dictionary(name: str = DEFAULT_DICT):
    try:
        return cv2.aruco.getPredefinedDictionary(_DICT_NAMES[name])
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown ArUco dictionary {name!r}; try one of {available_dictionaries()}"
        ) from exc


def make_detector(name: str = DEFAULT_DICT) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement helps homography accuracy a lot.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(get_dictionary(name), params)


def detect_markers(image: np.ndarray, dict_name: str = DEFAULT_DICT) -> list[DetectedMarker]:
    gray = to_gray(image)
    detector = make_detector(dict_name)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return []
    out: list[DetectedMarker] = []
    for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
        pts = marker_corners.reshape(4, 2).astype(float)
        out.append(
            DetectedMarker(
                marker_id=int(marker_id),
                corners=pts.tolist(),
                center=pts.mean(axis=0).tolist(),
            )
        )
    out.sort(key=lambda m: m.marker_id)
    return out


def generate_marker(
    marker_id: int, size_px: int = 200, dict_name: str = DEFAULT_DICT
) -> np.ndarray:
    dictionary = get_dictionary(dict_name)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
