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

# One identical id printed on every adhesive corner sticker (spec §6, §11.2).
# Distinct from the printed sheet's 0/1/2/3 so a mixed image is unambiguous.
# The sticker also carries an L-bracket + wedge graphic pointing at the page
# corner; the ArUco is a fixed physical size so the constellation scale gives a
# page-size estimate.
CORNER_STICKER_ID = 10
CORNER_STICKER_ARUCO_MM = 14.0  # printed ArUco side on a standard corner sticker

# Per-field metadata anchors (spec §11.3): one small ArUco immediately left of
# each metadata field's box. The id says *which* field the box is and its
# printed size gives the box's scale, so the detector no longer has to guess
# field boundaries from thin ruled lines. Ids 20-26, distinct from the corner
# markers (0-3) and the corner sticker (10).
METADATA_FIELD_IDS: dict[str, int] = {
    "document_id": 20,
    "page_id": 21,
    "topic_tags": 22,
    "left": 23,
    "above": 24,
    "below": 25,
    "right": 26,
}
FIELD_BY_MARKER_ID: dict[int, str] = {v: k for k, v in METADATA_FIELD_IDS.items()}
METADATA_FIELD_ANCHOR_MM = 8.0  # printed anchor side (~1/3 of an 24 mm corner)

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
