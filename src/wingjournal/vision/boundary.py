"""Low-level page-frame geometry (spec sections 23, 28-33).

Corner ordering, marker-role assignment, and the marker-constellation quad.
Hypothesis generation and scoring live in ``vision/hypothesis.py``; the
end-to-end boundary choice is ``vision/hypothesis.select_boundary``.
"""

from __future__ import annotations

import numpy as np

from wingjournal.models import DetectedMarker
from wingjournal.vision.aruco import MARKER_ROLE_IDS

# Canonical corner roles, TL -> TR -> BR -> BL (clockwise). Single source of
# truth for the whole package.
ROLES: tuple[str, ...] = tuple(MARKER_ROLE_IDS)

# Unit-ish direction of each role relative to the page centre.
ROLE_SIGNS: dict[str, tuple[float, float]] = {
    "TOP_LEFT": (-1.0, -1.0),
    "TOP_RIGHT": (1.0, -1.0),
    "BOTTOM_RIGHT": (1.0, 1.0),
    "BOTTOM_LEFT": (-1.0, 1.0),
}

_ID_TO_ROLE = {v: k for k, v in MARKER_ROLE_IDS.items()}


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""

    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)  # x + y
    diff = pts[:, 1] - pts[:, 0]  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def assign_roles_by_id(markers: list[DetectedMarker]) -> dict[str, DetectedMarker]:
    """Role per marker from the canonical id->corner convention (unambiguous).

    Empty unless every marker id is in the canonical set and no role repeats.
    """

    out: dict[str, DetectedMarker] = {}
    for m in markers:
        role = _ID_TO_ROLE.get(m.marker_id)
        if role is None or role in out:
            return {}
        out[role] = m
    return out


def assign_roles_greedy(markers: list[DetectedMarker]) -> dict[str, DetectedMarker]:
    """Assign each marker to at most one corner role (greedy, best score first).

    With 3 markers you get 3 roles, with 4+ you get 4. Never double-assigns.
    """

    if not markers:
        return {}
    centers = np.array([m.center for m in markers], dtype=np.float32)
    rel = centers - centers.mean(axis=0)
    pairs: list[tuple[float, int, str]] = []
    for i in range(len(markers)):
        for role, sign in ROLE_SIGNS.items():
            pairs.append((float(np.dot(rel[i], np.array(sign, np.float32))), i, role))
    pairs.sort(reverse=True)

    used_i: set[int] = set()
    used_role: set[str] = set()
    out: dict[str, DetectedMarker] = {}
    for _score, i, role in pairs:
        if i in used_i or role in used_role:
            continue
        used_i.add(i)
        used_role.add(role)
        out[role] = markers[i]
    return out


def best_roles(markers: list[DetectedMarker]) -> dict[str, DetectedMarker]:
    """Canonical-id role assignment when possible, else greedy geometry."""

    return assign_roles_by_id(markers) or assign_roles_greedy(markers)


def outer_corner_of(marker: DetectedMarker, page_centroid: np.ndarray) -> np.ndarray:
    corners = np.asarray(marker.corners, dtype=np.float32)
    dists = np.linalg.norm(corners - page_centroid, axis=1)
    return corners[int(np.argmax(dists))]


def complete_quad_from_three(points: dict[str, np.ndarray]) -> np.ndarray | None:
    """Given 3 of the 4 corner points, complete the parallelogram.

    Exact only for an affine view; under strong perspective the 4th corner is
    slightly off (see the eval harness's 3-marker bucket).
    """

    if len(points) != 3:
        return None
    missing = (set(ROLES) - set(points)).pop()
    opp = {
        "TOP_LEFT": ("TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"),
        "TOP_RIGHT": ("TOP_LEFT", "BOTTOM_RIGHT", "BOTTOM_LEFT"),
        "BOTTOM_RIGHT": ("TOP_RIGHT", "BOTTOM_LEFT", "TOP_LEFT"),
        "BOTTOM_LEFT": ("TOP_LEFT", "BOTTOM_RIGHT", "TOP_RIGHT"),
    }[missing]
    a, b, c = (points[r] for r in opp)  # missing = a + b - c
    points = dict(points)
    points[missing] = a + b - c
    return order_points(np.array([points[r] for r in ROLES], dtype=np.float32))


def quad_from_markers(markers: list[DetectedMarker]) -> np.ndarray | None:
    """Page quad from the *outer* corner of each of 4 corner markers."""

    if len(markers) < 4:
        return None
    roles = best_roles(markers)
    if len(roles) != 4:
        return None
    centroid = np.array([m.center for m in markers], dtype=np.float32).mean(axis=0)
    pts = np.array(
        [outer_corner_of(roles[r], centroid) for r in ROLES], dtype=np.float32
    )
    return order_points(pts)
