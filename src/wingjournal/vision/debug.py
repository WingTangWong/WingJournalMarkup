"""Debug overlays for the vision pipeline (M2 issue #14).

Each function returns a BGR image; :func:`write_overlays` dumps a numbered set.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from wingjournal.models import DetectedMarker, FiducialCandidate, Orientation, PageHypothesis

_GREEN = (0, 170, 0)
_BLUE = (200, 120, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 200, 220)


def _base(image: np.ndarray) -> np.ndarray:
    return image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def draw_markers(image: np.ndarray, markers: list[DetectedMarker]) -> np.ndarray:
    out = _base(image)
    for m in markers:
        pts = np.asarray(m.corners, dtype=np.int32)
        cv2.polylines(out, [pts], True, _GREEN, 3)
        cx, cy = (int(m.center[0]), int(m.center[1]))
        cv2.putText(out, f"id{m.marker_id}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN, 2, cv2.LINE_AA)
    return out


def draw_candidates(image: np.ndarray, candidates: list[FiducialCandidate]) -> np.ndarray:
    out = _base(image)
    for c in candidates:
        x, y, w, h = (int(v) for v in c.bbox)
        colour = _GREEN if c.decoded else _YELLOW
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        tag = c.inferred_role or "?"
        cv2.putText(out, f"{tag} {c.confidence:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


def draw_hypotheses(image: np.ndarray, hypotheses: list[PageHypothesis]) -> np.ndarray:
    out = _base(image)
    for i, h in enumerate(hypotheses):
        poly = np.asarray(h.polygon, dtype=np.int32)
        colour = _RED if i == 0 else _BLUE
        cv2.polylines(out, [poly], True, colour, 3 if i == 0 else 1)
        p0 = poly[0]
        cv2.putText(out, f"{h.source} {h.score:.2f}", (int(p0[0]) + 5, int(p0[1]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return out


def draw_chosen(
    image: np.ndarray, polygon: list[list[float]], orientation: Orientation
) -> np.ndarray:
    out = _base(image)
    poly = np.asarray(polygon, dtype=np.int32)
    cv2.polylines(out, [poly], True, _RED, 4)
    for label, pt in zip("ABCD", poly, strict=True):
        cv2.circle(out, (int(pt[0]), int(pt[1])), 8, _RED, -1)
        cv2.putText(out, label, (int(pt[0]) + 10, int(pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, _RED, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"orientation {orientation.degrees} deg via {orientation.method} "
        f"({orientation.confidence:.2f})",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2, cv2.LINE_AA,
    )
    return out


def write_overlays(
    out_dir: str | Path,
    name: str,
    image: np.ndarray,
    markers: list[DetectedMarker],
    candidates: list[FiducialCandidate],
    hypotheses: list[PageHypothesis],
    polygon: list[list[float]],
    orientation: Orientation,
) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    layers = {
        "01_markers": draw_markers(image, markers),
        "02_candidates": draw_candidates(image, candidates),
        "03_hypotheses": draw_hypotheses(image, hypotheses),
        "04_chosen": draw_chosen(image, polygon, orientation),
    }
    paths = []
    for suffix, img in layers.items():
        p = d / f"{name}.{suffix}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths
