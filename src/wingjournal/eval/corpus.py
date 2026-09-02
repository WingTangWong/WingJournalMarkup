"""Labelled synthetic corpus for boundary + orientation evaluation.

Each case is a scene (a synthetic WJM page placed under a known perspective,
in-plane rotation, and marker dropout) plus ground truth: the true page
quadrilateral and the true upright-correction angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wingjournal.vision.boundary import ROLES, order_points
from wingjournal.vision.synthetic import make_page, marker_regions, warp_page

_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class EvalCase:
    name: str
    image: np.ndarray
    true_polygon: list[list[float]]  # scene coords, ordered TL, TR, BR, BL
    true_orientation: int  # degrees CW to apply to make the page upright
    dropped_markers: int  # markers that are missing or reduced to blank squares
    perspective: str  # "flat" | "mild" | "strong"


def _add_noise(scene: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return scene
    noisy = scene.astype(np.float32) + rng.normal(0, sigma, scene.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(noisy, (3, 3), 0)


def make_case(
    name: str,
    rng: np.random.Generator,
    rotation_cw: int = 0,
    n_drop: int = 0,
    perspective: str = "mild",
    noise_sigma: float = 4.0,
    blank: bool = True,
) -> EvalCase:
    degraded = tuple(
        ROLES[i] for i in rng.choice(4, size=n_drop, replace=False)
    ) if n_drop else ()
    if blank:
        page = make_page(blank_roles=degraded)
    else:
        page = make_page(drop_roles=degraded)

    if _ROTATIONS[rotation_cw] is not None:
        page = cv2.rotate(page, _ROTATIONS[rotation_cw])

    # Canvas comfortably fits the page in either orientation.
    long_side = max(page.shape[:2])
    ch = cw = round(long_side * 1.45)
    canvas = (ch, cw)
    if perspective == "flat":
        h, w = page.shape[:2]
        ox, oy = (cw - w) // 2, (ch - h) // 2
        quad = np.array(
            [[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]], dtype=np.float32
        )
    else:
        mag = {"mild": 0.04, "strong": 0.09}[perspective]
        base = np.array(
            [[cw * 0.14, ch * 0.12], [cw * 0.86, ch * 0.15],
             [cw * 0.84, ch * 0.88], [cw * 0.13, ch * 0.85]],
            dtype=np.float32,
        )
        quad = (base + rng.uniform(-mag, mag, (4, 2)) * np.array([cw, ch])).astype(np.float32)

    scene, _dst_page_quad = warp_page(page, canvas=canvas, quad=quad)
    scene = _add_noise(scene, rng, noise_sigma)

    # Ground-truth boundary is the marker *constellation* frame (the outer corner
    # of each corner marker), which is what the spec's coordinate system is
    # pinned to - not the physical paper edge.
    ph, pw = page.shape[:2]
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, quad)
    reg = marker_regions(pw, ph)
    outer = np.array(
        [
            [reg["TOP_LEFT"][0], reg["TOP_LEFT"][1]],
            [reg["TOP_RIGHT"][0] + reg["TOP_RIGHT"][2], reg["TOP_RIGHT"][1]],
            [reg["BOTTOM_RIGHT"][0] + reg["BOTTOM_RIGHT"][2],
             reg["BOTTOM_RIGHT"][1] + reg["BOTTOM_RIGHT"][3]],
            [reg["BOTTOM_LEFT"][0], reg["BOTTOM_LEFT"][1] + reg["BOTTOM_LEFT"][3]],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    frame = cv2.perspectiveTransform(outer, homography).reshape(-1, 2)

    return EvalCase(
        name=name,
        image=scene,
        true_polygon=order_points(frame).tolist(),
        true_orientation=(360 - rotation_cw) % 360,
        dropped_markers=n_drop,
        perspective=perspective,
    )


def generate_corpus(n: int = 24, seed: int = 0) -> list[EvalCase]:
    rng = np.random.default_rng(seed)
    perspectives = ["flat", "mild", "mild", "strong"]
    rotations = [0, 0, 90, 180, 270]
    cases: list[EvalCase] = []
    for i in range(n):
        rot = rotations[i % len(rotations)]
        persp = perspectives[i % len(perspectives)]
        # weight the corpus toward the well-marked cases
        n_drop = (0, 0, 0, 1, 1, 2, 3)[i % 7]
        cases.append(
            make_case(
                f"case{i:03d}", rng, rotation_cw=rot, n_drop=n_drop,
                perspective=persp, blank=(i % 5 != 0),
            )
        )
    return cases
