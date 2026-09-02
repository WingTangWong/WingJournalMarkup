"""Synthetic WJM pages for testing and demos.

Not part of the processing pipeline - this is a generator used by
``wingjournal make-test-page`` and the test-suite so we can exercise the
detector/rectifier without real scans.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.vision.aruco import DEFAULT_DICT, generate_marker
from wingjournal.vision.boundary import ROLES

DEFAULT_IDS = (0, 1, 2, 3)  # positional: ROLES[i] -> DEFAULT_IDS[i]


def marker_regions(
    width: int = 1000, height: int = 1400, margin: int = 48, marker_px: int = 130
) -> dict[str, tuple[int, int, int, int]]:
    """role -> (x, y, w, h) of each corner marker on a flat page."""

    return {
        "TOP_LEFT": (margin, margin, marker_px, marker_px),
        "TOP_RIGHT": (width - margin - marker_px, margin, marker_px, marker_px),
        "BOTTOM_RIGHT": (
            width - margin - marker_px, height - margin - marker_px, marker_px, marker_px
        ),
        "BOTTOM_LEFT": (margin, height - margin - marker_px, marker_px, marker_px),
    }


def make_page(
    width: int = 1000,
    height: int = 1400,
    margin: int = 48,
    marker_px: int = 130,
    marker_ids: tuple[int, int, int, int] = DEFAULT_IDS,
    dict_name: str = DEFAULT_DICT,
    drop_roles: tuple[str, ...] = (),
    blank_roles: tuple[str, ...] = (),
    literal_box: bool = False,
) -> np.ndarray:
    """A white page with four corner ArUco markers and some mock content.

    ``drop_roles`` omits markers entirely; ``blank_roles`` replaces them with an
    empty square outline (a blank / damaged sticker - still geometric evidence,
    spec section 26). ``literal_box`` adds an escaped image region (spec §16).
    """

    page = np.full((height, width), 255, dtype=np.uint8)

    regions = marker_regions(width, height, margin, marker_px)
    for role, mid in zip(ROLES, marker_ids, strict=True):
        x, y, w, h = regions[role]
        if role in drop_roles:
            continue
        if role in blank_roles:
            cv2.rectangle(page, (x, y), (x + w, y + h), 0, max(3, marker_px // 20))
            continue
        page[y : y + h, x : x + w] = generate_marker(mid, marker_px, dict_name)

    # Mock metadata block (spec section 11).
    top = margin + marker_px + 40
    cv2.rectangle(page, (margin + 30, top), (width - margin - 30, top + 150), 0, 3)
    cv2.line(page, (margin + 30, top + 75), (width - margin - 30, top + 75), 0, 2)
    cv2.putText(page, "#Research  #P017  #AI", (margin + 50, top + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    cv2.putText(page, "#P016   #P027   #P018", (margin + 50, top + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)

    # Mock bullets + a node box.
    by = top + 230
    for label in ("- investigate embedding methods", "x install dependencies"):
        cv2.putText(page, label, (margin + 40, by), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        by += 55
    cv2.rectangle(page, (margin + 40, by + 20), (margin + 460, by + 200), 0, 3)
    cv2.putText(page, "Vector Database", (margin + 60, by + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)

    if literal_box:
        lx, ly = margin + 40, by + 250
        lw, lh = width - 2 * margin - 80, height - ly - margin - marker_px - 40
        draw_literal_box(page, lx, ly, lw, lh)

    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def draw_literal_box(page: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """A rectangle with solid diagonal black fills in all four corners (spec §16)."""

    cv2.rectangle(page, (x, y), (x + w, y + h), 0, 2)
    t = max(16, int(min(w, h) * 0.22))
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1),
                           (x + w, y + h, -1, -1), (x, y + h, 1, -1)):
        cv2.fillConvexPoly(
            page,
            np.array([[cx, cy], [cx + dx * t, cy], [cx, cy + dy * t]], np.int32),
            0,
        )
    # some "freehand" scribble inside that must never be parsed
    cv2.line(page, (x + w // 4, y + h // 2), (x + 3 * w // 4, y + h // 3), 0, 3)
    cv2.circle(page, (x + w // 2, y + 2 * h // 3), h // 8, 0, 3)


def warp_page(
    page: np.ndarray,
    canvas: tuple[int, int] = (1600, 1200),
    quad: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place ``page`` onto a larger canvas under a random perspective.

    Returns ``(scene, dst_quad)`` where ``dst_quad`` is the true page
    boundary in the scene, ordered TL, TR, BR, BL.
    """

    ch, cw = canvas
    h, w = page.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    if quad is None:
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.08, 0.08, size=(4, 2)) * np.array([cw, ch])
        base = np.array(
            [[cw * 0.18, ch * 0.12], [cw * 0.85, ch * 0.16],
             [cw * 0.82, ch * 0.9], [cw * 0.15, ch * 0.86]],
            dtype=np.float32,
        )
        quad = (base + jitter).astype(np.float32)

    homography = cv2.getPerspectiveTransform(src, quad)
    scene = np.full((ch, cw, 3), 235, dtype=np.uint8)
    warped = cv2.warpPerspective(page, homography, (cw, ch), borderValue=(235, 235, 235))
    mask = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), homography, (cw, ch)
    )
    scene[mask > 0] = warped[mask > 0]
    return scene, quad
