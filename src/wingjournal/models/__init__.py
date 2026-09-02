"""Core WJM data models.

Only the subset needed for the ingestion pipeline (through perspective
normalization) is implemented so far. See docs/ROADMAP.md for the rest.

These are plain dataclasses; serialize with :func:`dataclasses.asdict`.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from dataclasses import dataclass, field


def _new_uuid() -> str:
    return str(_uuid.uuid4())


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass
class DetectedMarker:
    """A decoded ArUco marker."""

    marker_id: int
    corners: list[list[float]]  # 4 x [x, y], clockwise from marker top-left
    center: list[float]


@dataclass
class FiducialCandidate:
    """A marker-like region that may or may not be decodable.

    Mirrors section 9 / 26 of the spec: an undecoded square in a corner is
    still geometric evidence for the page frame.
    """

    bbox: list[float]  # [x, y, w, h]
    center: list[float]
    decoded: bool = False
    marker_id: int | None = None
    inferred_role: str | None = None  # TOP_LEFT / TOP_RIGHT / BOTTOM_RIGHT / BOTTOM_LEFT
    reason: str = ""
    confidence: float = 0.0


@dataclass
class PageBoundary:
    polygon: list[list[float]]  # 4 x [x, y] ordered TL, TR, BR, BL
    # method mirrors PageHypothesis.source (the winning hypothesis)
    method: str
    confidence: float


@dataclass
class PageHypothesis:
    """A candidate page frame plus the evidence behind it (spec sections 30-31)."""

    polygon: list[list[float]]  # 4 x [x, y], ordered TL, TR, BR, BL
    # source: aruco_constellation | aruco_partial | aruco_three_corner |
    #         square_candidates | largest_quad | full_frame
    source: str
    decoded_fiducials: int = 0
    inferred_fiducials: int = 0
    evidence: dict[str, float] = field(default_factory=dict)  # signal -> [0, 1]
    penalties: dict[str, float] = field(default_factory=dict)  # signal -> [0, 1]
    score: float = 0.0


@dataclass
class Orientation:
    degrees: int  # 0 / 90 / 180 / 270, clockwise rotation to make the page upright
    method: str  # "aruco_ids" | "text_baseline" | "assumed"
    confidence: float
    # True when the method cannot tell 0 from 180 (or 90 from 270): the axis is
    # known but not which way is up. `degrees` is then the better of the two.
    flip_ambiguous: bool = False


@dataclass
class Capture:
    """One image observation of a page (spec sections 3.4, 43)."""

    uuid: str = field(default_factory=_new_uuid)
    page_uuid: str | None = None
    timestamp: str = field(default_factory=_now)
    source_type: str = "file"

    raw_image_path: str | None = None
    normalized_image_path: str | None = None

    page_boundary_method: str | None = None
    page_boundary_confidence: float = 0.0
    page_boundary_polygon: list[list[float]] | None = None  # raw-image coords, TL,TR,BR,BL

    orientation_degrees: int | None = None
    orientation_confidence: float = 0.0
    orientation_method: str | None = None
    orientation_flip_ambiguous: bool = False

    # maps raw-image coords -> the saved normalized image (rotation folded in)
    homography: list[list[float]] | None = None

    page_hypotheses: list[PageHypothesis] = field(default_factory=list)

    detected_fiducials: list[DetectedMarker] = field(default_factory=list)
    inferred_fiducials: list[FiducialCandidate] = field(default_factory=list)
    detected_elements: list[dict] = field(default_factory=list)

    previous_capture_uuid: str | None = None

    notes: list[str] = field(default_factory=list)
