"""Multi-evidence page-boundary hypotheses and scoring (spec sections 30-31).

Signals are deliberately few; weights live in :class:`ScoringWeights` and can be
loaded from a JSON config so they can be tuned against the evaluation corpus
rather than hand-set forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from wingjournal.models import (
    DetectedMarker,
    FiducialCandidate,
    PageBoundary,
    PageHypothesis,
)
from wingjournal.vision.boundary import (
    ROLE_SIGNS,
    ROLES,
    best_roles,
    complete_quad_from_three,
    order_points,
    outer_corner_of,
    quad_from_markers,
)
from wingjournal.vision.envelope import envelope_hypotheses
from wingjournal.vision.fiducial_candidates import find_square_candidates, refine_squares
from wingjournal.vision.preprocess import Preprocessed, find_quads

# Plausible page aspect ratios (long side / short side).
_PAGE_ASPECTS = (279.4 / 215.9, 297.0 / 210.0)  # letter, A4


@dataclass(frozen=True)
class ScoringWeights:
    decoded_markers: float = 0.32
    rectangularity: float = 0.22
    content_containment: float = 0.28
    aspect_plausible: float = 0.10
    area_plausible: float = 0.08
    # penalties (subtracted)
    penalty_not_convex: float = 0.5
    penalty_clipping: float = 0.4

    @classmethod
    def load(cls, path: str | Path) -> ScoringWeights:
        data = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown scoring weight(s): {sorted(unknown)}")
        return replace(cls(), **data)

    def evidence_total(self) -> float:
        return (
            self.decoded_markers
            + self.rectangularity
            + self.content_containment
            + self.aspect_plausible
            + self.area_plausible
        )


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #
def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def rectangularity(polygon: np.ndarray) -> float:
    p = polygon.astype(np.float32)
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 1e-3:
        return 0.0
    opp = min(lengths[0], lengths[2]) / max(lengths[0], lengths[2])
    opp *= min(lengths[1], lengths[3]) / max(lengths[1], lengths[3])
    units = edges / lengths[:, None]
    angles = [
        abs(np.dot(units[i], -units[(i - 1) % 4])) for i in range(4)
    ]  # 0 when perpendicular
    squareness = 1.0 - float(np.mean(angles))
    return float(max(0.0, opp) * max(0.0, squareness))


def content_containment(binary: np.ndarray, polygon: np.ndarray) -> float:
    total = int((binary > 0).sum())
    if total == 0:
        return 1.0
    mask = _polygon_mask(binary.shape, polygon)
    inside = int(((binary > 0) & (mask > 0)).sum())
    return inside / total


def aspect_plausible(polygon: np.ndarray) -> float:
    p = polygon.astype(np.float32)
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    if min(w, h) < 1e-3:
        return 0.0
    ratio = max(w, h) / min(w, h)
    best = min(abs(ratio - a) for a in _PAGE_ASPECTS)
    return float(np.exp(-((best / 0.25) ** 2)))


def area_plausible(shape: tuple[int, int], polygon: np.ndarray) -> float:
    h, w = shape[:2]
    frac = cv2.contourArea(polygon.astype(np.float32)) / (h * w)
    if 0.2 <= frac <= 0.98:
        return 1.0
    if frac < 0.2:
        return float(max(0.0, frac / 0.2))
    return float(max(0.0, (1.05 - frac) / 0.07))


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _square_corner(sq: FiducialCandidate, role: str) -> np.ndarray:
    x, y, bw, bh = sq.bbox
    return np.array(
        {
            "TOP_LEFT": (x, y),
            "TOP_RIGHT": (x + bw, y),
            "BOTTOM_RIGHT": (x + bw, y + bh),
            "BOTTOM_LEFT": (x, y + bh),
        }[role],
        dtype=np.float32,
    )


def _corner_points(
    markers: list[DetectedMarker], squares: list[FiducialCandidate]
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Outer-corner point per role, decoded markers first then square candidates.

    Square candidates fill only the roles markers left open, assigned greedily by
    direction from the marker-established page centre (or the square cloud when
    there are no markers). Returns ``(points, marker_roles)``.
    """

    points: dict[str, np.ndarray] = {}
    marker_roles: set[str] = set()
    if markers:
        centroid = np.array([m.center for m in markers], dtype=np.float32).mean(axis=0)
        for role, m in best_roles(markers).items():
            points[role] = outer_corner_of(m, centroid)
            marker_roles.add(role)

    open_roles = [r for r in ROLES if r not in points]
    if squares and open_roles:
        if points:
            anchor = np.mean(list(points.values()), axis=0)
        else:
            anchor = np.mean([s.center for s in squares], axis=0)
        ranked = sorted(
            (
                (float(np.dot(np.asarray(s.center) - anchor, ROLE_SIGNS[r])), i, r)
                for i, s in enumerate(squares)
                for r in open_roles
            ),
            reverse=True,
        )
        used_sq: set[int] = set()
        for dot, i, role in ranked:
            if dot <= 0 or i in used_sq or role in points:
                continue
            points[role] = _square_corner(squares[i], role)
            used_sq.add(i)
    return points, marker_roles


def generate_hypotheses(
    pre: Preprocessed,
    markers: list[DetectedMarker],
    squares: list[FiducialCandidate] | None = None,
    sticker_quad: np.ndarray | None = None,
) -> list[PageHypothesis]:
    squares = squares or []
    h, w = pre.gray.shape[:2]
    out: list[PageHypothesis] = []

    # 0. adhesive corner stickers: wedge tips are the page corners (spec §11.2)
    if sticker_quad is not None:
        out.append(
            PageHypothesis(
                polygon=order_points(np.asarray(sticker_quad, dtype=np.float32)).tolist(),
                source="corner_stickers",
                decoded_fiducials=len(sticker_quad),
            )
        )

    # 1. full marker constellation (outer corners)
    quad = quad_from_markers(markers)
    if quad is not None:
        out.append(
            PageHypothesis(
                polygon=quad.tolist(),
                source="aruco_constellation",
                decoded_fiducials=len(markers),
            )
        )

    points, marker_roles = _corner_points(markers, squares)
    n_markers = len(marker_roles)

    # 2. four corners from a marker/square mix (step 1 already covered 4 markers)
    if len(points) == 4 and n_markers < 4:
        poly = order_points(np.array([points[r] for r in ROLES], dtype=np.float32))
        out.append(
            PageHypothesis(
                polygon=poly.tolist(),
                source="aruco_partial" if n_markers else "square_candidates",
                decoded_fiducials=n_markers,
                inferred_fiducials=4 - n_markers,
            )
        )

    # 3. three corners -> complete the parallelogram (spec section 32 tier C).
    #    From 3 decoded markers even if a (weaker) square could fill the 4th, so
    #    the scorer can choose; otherwise from any 3 combined points.
    three = None
    if n_markers == 3:
        three = {r: points[r] for r in marker_roles}
    elif len(points) == 3:
        three = dict(points)
    if three is not None:
        poly = complete_quad_from_three(three)
        if poly is not None:
            out.append(
                PageHypothesis(
                    polygon=poly.tolist(),
                    source="three_corner",
                    decoded_fiducials=min(n_markers, 3),
                    inferred_fiducials=4 - min(n_markers, 3),
                )
            )

    # 4. largest convex quad in the edge map
    for cand in find_quads(pre.edges)[:2]:
        out.append(
            PageHypothesis(
                polygon=order_points(cand).tolist(), source="largest_quad"
            )
        )

    # 5. content-anchored envelope (spec sections 27-28, orientation tier G)
    out.extend(envelope_hypotheses(pre))

    # 6. always: the whole frame
    out.append(
        PageHypothesis(
            polygon=[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            source="full_frame",
        )
    )
    return out


def score_hypothesis(
    hyp: PageHypothesis, pre: Preprocessed, weights: ScoringWeights
) -> PageHypothesis:
    poly = np.array(hyp.polygon, dtype=np.float32)
    shape = pre.gray.shape[:2]

    ev = {
        "decoded_markers": min(1.0, hyp.decoded_fiducials / 4.0),
        "rectangularity": rectangularity(poly),
        "content_containment": content_containment(pre.binary, poly),
        "aspect_plausible": aspect_plausible(poly),
        "area_plausible": area_plausible(shape, poly),
    }
    pen = {
        "not_convex": 0.0 if cv2.isContourConvex(poly.astype(np.int32)) else 1.0,
        "clipping": max(0.0, 1.0 - ev["content_containment"]),
    }

    evidence = (
        weights.decoded_markers * ev["decoded_markers"]
        + weights.rectangularity * ev["rectangularity"]
        + weights.content_containment * ev["content_containment"]
        + weights.aspect_plausible * ev["aspect_plausible"]
        + weights.area_plausible * ev["area_plausible"]
    ) / weights.evidence_total()
    penalty = (
        weights.penalty_not_convex * pen["not_convex"]
        + weights.penalty_clipping * pen["clipping"]
    )
    hyp.evidence = {k: round(v, 4) for k, v in ev.items()}
    hyp.penalties = {k: round(v, 4) for k, v in pen.items()}
    hyp.score = round(evidence - penalty, 4)
    return hyp


def rank_hypotheses(
    pre: Preprocessed,
    markers: list[DetectedMarker],
    squares: list[FiducialCandidate] | None = None,
    weights: ScoringWeights | None = None,
    sticker_quad: np.ndarray | None = None,
) -> list[PageHypothesis]:
    weights = weights or ScoringWeights()
    scored = [
        score_hypothesis(h, pre, weights)
        for h in generate_hypotheses(pre, markers, squares, sticker_quad)
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored


def _corner_shift(a: list[list[float]], b: list[list[float]]) -> float:
    aa, bb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(aa - bb, axis=1).mean())


def select_boundary(
    pre: Preprocessed,
    markers: list[DetectedMarker],
    squares: list[FiducialCandidate] | None = None,
    weights: ScoringWeights | None = None,
    max_passes: int = 3,
    sticker_quad: np.ndarray | None = None,
) -> tuple[PageBoundary, list[PageHypothesis], list[FiducialCandidate]]:
    """Iteratively rank hypotheses (spec section 34) and return the winner.

    Each pass re-detects corner squares near the current best frame; it stops
    when a pass no longer improves the score or the frame stops moving.
    ``sticker_quad`` is the page quad from adhesive corner stickers (spec §11.2).
    """

    if squares is None:
        squares = find_square_candidates(pre, exclude=markers)

    ranked = rank_hypotheses(pre, markers, squares, weights, sticker_quad)
    best = ranked[0]
    h, w = pre.gray.shape[:2]
    converge = 0.01 * float(np.hypot(w, h))

    for _ in range(max(0, max_passes - 1)):
        refined = refine_squares(pre, best.polygon, exclude=markers)
        if not refined:
            break
        merged = _merge_squares(squares, refined)
        ranked2 = rank_hypotheses(pre, markers, merged, weights, sticker_quad)
        if ranked2[0].score <= best.score + 1e-4:
            break
        shift = _corner_shift(ranked2[0].polygon, best.polygon)
        best, ranked, squares = ranked2[0], ranked2, merged
        if shift < converge:
            break

    confidence = round(float(np.clip(best.score, 0.0, 1.0)), 3)
    boundary = PageBoundary(polygon=best.polygon, method=best.source, confidence=confidence)
    return boundary, ranked, squares


def _merge_squares(
    base: list[FiducialCandidate], extra: list[FiducialCandidate]
) -> list[FiducialCandidate]:
    out = list(base)
    for e in extra:
        ec = np.asarray(e.center, dtype=np.float32)
        side = max(e.bbox[2], e.bbox[3])
        if all(np.linalg.norm(np.asarray(b.center, np.float32) - ec) > 0.5 * side for b in out):
            out.append(e)
    return out
