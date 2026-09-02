"""The ingestion pipeline.

    acquire -> preprocess -> detect ArUco + square candidates
    -> ranked page-boundary hypotheses -> orientation resolution
    -> perspective normalization + upright rotation -> Capture record (+ JSON)

Downstream stages (literal-box masking, OCR, markup parsing, graph update,
capture reconciliation) are stubs on the roadmap.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from wingjournal.capture import CaptureSource, source_for
from wingjournal.models import (
    Capture,
    DetectedMarker,
    FiducialCandidate,
    Orientation,
    PageBoundary,
    PageHypothesis,
)
from wingjournal.vision.aruco import DEFAULT_DICT, detect_markers
from wingjournal.vision.boundary import best_roles
from wingjournal.vision.hypothesis import ScoringWeights, select_boundary
from wingjournal.vision.orientation import resolve_orientation
from wingjournal.vision.preprocess import preprocess
from wingjournal.vision.rectify import rectify


@dataclass
class IngestResult:
    name: str
    capture: Capture
    normalized_image: np.ndarray
    orientation: Orientation
    hypotheses: list[PageHypothesis] = field(default_factory=list)
    square_candidates: list[FiducialCandidate] = field(default_factory=list)


def _decoded_candidates(
    markers: list[DetectedMarker], boundary: PageBoundary
) -> list[FiducialCandidate]:
    role_by_id = {m.marker_id: r for r, m in best_roles(markers).items()}
    out: list[FiducialCandidate] = []
    for m in markers:
        x, y, w, h = cv2.boundingRect(np.asarray(m.corners, dtype=np.float32))
        out.append(
            FiducialCandidate(
                bbox=[float(x), float(y), float(w), float(h)],
                center=m.center,
                decoded=True,
                marker_id=m.marker_id,
                inferred_role=role_by_id.get(m.marker_id),
                reason="decoded_aruco",
                confidence=min(1.0, boundary.confidence + 0.03),
            )
        )
    return out


def ingest_image(
    name: str,
    image: np.ndarray,
    dict_name: str = DEFAULT_DICT,
    source_type: str = "file",
    raw_path: str | None = None,
    weights: ScoringWeights | None = None,
) -> IngestResult:
    pre = preprocess(image)
    markers = detect_markers(pre.gray, dict_name)

    boundary, hypotheses, squares = select_boundary(pre, markers, weights=weights)
    orientation = resolve_orientation(pre, markers, boundary.polygon)

    quad = np.asarray(boundary.polygon, dtype=np.float32)
    normalized, homography = rectify(image, quad, rotate_degrees=orientation.degrees)

    capture = Capture(
        source_type=source_type,
        raw_image_path=raw_path,
        page_boundary_method=boundary.method,
        page_boundary_confidence=boundary.confidence,
        page_boundary_polygon=boundary.polygon,
        orientation_degrees=orientation.degrees,
        orientation_confidence=orientation.confidence,
        orientation_method=orientation.method,
        orientation_flip_ambiguous=orientation.flip_ambiguous,
        homography=homography.tolist(),
        detected_fiducials=markers,
        inferred_fiducials=_decoded_candidates(markers, boundary) + squares,
        page_hypotheses=hypotheses,
    )
    capture.notes.append(
        f"{len(markers)} ArUco marker(s), {len(squares)} square candidate(s); "
        f"boundary via {boundary.method} (score {boundary.confidence:.2f}); "
        f"orientation {orientation.degrees} deg via {orientation.method}"
        + (" (flip ambiguous)" if orientation.flip_ambiguous else "")
    )
    if not markers and not squares:
        capture.notes.append(
            "no fiducial evidence - boundary is a geometric guess (spec sections 27-28)"
        )
    return IngestResult(
        name=name,
        capture=capture,
        normalized_image=normalized,
        orientation=orientation,
        hypotheses=hypotheses,
        square_candidates=squares,
    )


def ingest_path(
    path: str | Path,
    out_dir: str | Path,
    dict_name: str = DEFAULT_DICT,
    recursive: bool = False,
    weights: ScoringWeights | None = None,
    debug: bool = False,
) -> list[IngestResult]:
    src: CaptureSource = source_for(path, recursive=recursive)
    out_dir = Path(out_dir)
    (out_dir / "normalized").mkdir(parents=True, exist_ok=True)
    (out_dir / "captures").mkdir(parents=True, exist_ok=True)

    results: list[IngestResult] = []
    for name, image in src:
        result = ingest_image(
            name, image, dict_name=dict_name, source_type=src.source_type,
            raw_path=str(path), weights=weights,
        )
        norm_path = out_dir / "normalized" / f"{name}.png"
        cv2.imwrite(str(norm_path), result.normalized_image)
        result.capture.normalized_image_path = str(norm_path)

        cap_path = out_dir / "captures" / f"{name}.json"
        cap_path.write_text(json.dumps(dataclasses.asdict(result.capture), indent=2))

        if debug:
            from wingjournal.vision.debug import write_overlays

            write_overlays(
                out_dir / "debug", name, image,
                result.capture.detected_fiducials,
                result.square_candidates,
                result.hypotheses,
                result.capture.page_boundary_polygon,
                result.orientation,
            )
        results.append(result)
    return results
