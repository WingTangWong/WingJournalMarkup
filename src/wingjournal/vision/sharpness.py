"""Capture sharpness assessment (spec §9.x — quality gate).

A photograph that is soft — motion blur, missed focus, too far — rectifies into a
page whose thin ink is unreadable. We score sharpness two ways and combine them:

* **global** — variance of the Laplacian over the page (the classic focus metric);
* **targeted** — edge acutance measured *at the known fiducials* (the corner ArUco
  markers and the metadata-block registration marks). Their edges are a known
  step, so a soft edge there is a soft photo, independent of page content.

The targeted score is the important one: it flags exactly the marks the detector
and OCR depend on. The live capture app uses ``report.blurry`` as a hard gate for
auto-shutter; ``ingest`` records the whole report on the ``Capture``.

Score the *grab*, not the rectified page (spec §9.1): ``warpPerspective``'s cubic
upscale smooths every edge and makes a soft photo look passable, so
:func:`assess_capture` measures the deciding sharpness on the raw capture at the
fiducials, before rectify. The rectified probes are still reported for detail but
they do not flip ``blurry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from wingjournal.vision.preprocess import to_gray
from wingjournal.vision.registration import RegistrationMark, _edge_acutance

# Laplacian variance mapped through this: <= FLOOR reads as fully soft, >= CEIL
# as fully sharp. Tuned for ~1600px-long normalized pages.
_LAPVAR_FLOOR = 25.0
_LAPVAR_CEIL = 320.0

# a probe below this acutance is "not sharp"
MIN_ACUTANCE = 0.30
# overall score below this (0..1) is "blurry"
MIN_SCORE = 0.45


@dataclass
class SharpnessProbe:
    name: str          # e.g. "marker:TOP_LEFT", "registration:0"
    acutance: float
    sharp: bool


@dataclass
class SharpnessReport:
    score: float                          # 0 (soft) .. 1 (crisp)
    laplacian_variance: float
    global_score: float
    probe_score: float
    probes: list[SharpnessProbe] = field(default_factory=list)
    blurry: bool = False
    # set by assess_capture: the same score/verdict on the de-warped page, for
    # reference only (the raw-grab read above is what gates a capture)
    rectified_score: float | None = None
    rectified_blurry: bool | None = None

    def summary(self) -> str:
        soft = [p.name for p in self.probes if not p.sharp]
        tail = f"; soft at {', '.join(soft)}" if soft else ""
        rect = "" if self.rectified_score is None else f", rectified {self.rectified_score:.2f}"
        return (
            f"sharpness {self.score:.2f} (lapvar {self.laplacian_variance:.0f}{rect}){tail}"
        )


def laplacian_variance(gray: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> float:
    g = to_gray(gray)
    if roi is not None:
        x, y, w, h = roi
        g = g[y:y + h, x:x + w]
    if g.size == 0:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _marker_acutance(gray: np.ndarray, corners: list[list[float]]) -> float:
    pts = np.asarray(corners, dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    side = float(np.linalg.norm(pts[0] - pts[1]))
    return _edge_acutance(gray, cx, cy, side)


def assess(
    image: np.ndarray,
    markers=None,
    registration_marks: list[RegistrationMark] | None = None,
    min_acutance: float = MIN_ACUTANCE,
    min_score: float = MIN_SCORE,
) -> SharpnessReport:
    """Score ``image`` (a normalized/rectified page). ``markers`` are
    :class:`DetectedMarker`-like (``.corners``, optional ``.marker_id``)."""

    gray = to_gray(image)
    lapvar = laplacian_variance(gray)
    global_score = float(np.clip(
        (lapvar - _LAPVAR_FLOOR) / (_LAPVAR_CEIL - _LAPVAR_FLOOR), 0.0, 1.0
    ))

    probes: list[SharpnessProbe] = []
    for m in markers or []:
        a = _marker_acutance(gray, m.corners)
        name = f"marker:{getattr(m, 'marker_id', '?')}"
        probes.append(SharpnessProbe(name, round(a, 3), a >= min_acutance))
    for i, rm in enumerate(registration_marks or []):
        probes.append(SharpnessProbe(f"registration:{i}", rm.acutance, rm.acutance >= min_acutance))

    probe_score = (
        float(np.mean([p.acutance for p in probes])) if probes else global_score
    )
    # weight the targeted probes; fall back to global when we have none
    score = probe_score if probes else global_score
    if probes:
        score = round(0.7 * probe_score + 0.3 * global_score, 3)

    blurry = score < min_score or any(not p.sharp for p in probes)

    return SharpnessReport(
        score=round(score, 3),
        laplacian_variance=round(lapvar, 1),
        global_score=round(global_score, 3),
        probe_score=round(probe_score, 3),
        probes=probes,
        blurry=blurry,
    )


def assess_capture(
    source_image: np.ndarray,
    source_markers=None,
    rectified_image: np.ndarray | None = None,
    rectified_markers=None,
    registration_marks: list[RegistrationMark] | None = None,
    min_acutance: float = MIN_ACUTANCE,
    min_score: float = MIN_SCORE,
) -> SharpnessReport:
    """Score the capture the way a live app must (spec §9.1).

    The verdict comes from ``source_image`` at ``source_markers`` — the raw grab,
    before any perspective upscale. When a rectified page is supplied its probes
    are appended (prefixed ``rectified:``) and its score recorded on
    ``rectified_score`` / ``rectified_blurry``, but they never flip ``blurry``.
    """

    shot = assess(source_image, source_markers, None, min_acutance, min_score)
    if rectified_image is None:
        return shot

    rect = assess(
        rectified_image, rectified_markers, registration_marks, min_acutance, min_score
    )
    shot.probes = [
        SharpnessProbe(f"shot:{p.name}", p.acutance, p.sharp) for p in shot.probes
    ] + [
        SharpnessProbe(
            p.name if p.name.startswith("registration:") else f"rectified:{p.name}",
            p.acutance,
            p.sharp,
        )
        for p in rect.probes
    ]
    shot.rectified_score = rect.score
    shot.rectified_blurry = rect.blurry
    return shot
