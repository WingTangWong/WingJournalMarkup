"""The ingestion pipeline.

    acquire -> preprocess -> detect ArUco + square candidates
    -> iterative ranked page-boundary hypotheses -> orientation resolution
    -> perspective normalization (fixed coords, upright rotation folded into the
       homography) -> literal-region detect + mask -> metadata-block detect
    -> (optional) OCR the metadata cells and, with parse_body, the page body
    -> Capture record -> optional persist to a store

Diagram-graph extraction and capture reconciliation are still on the roadmap
(M6, M8).
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
from wingjournal.recognition.metadata_block import detect_metadata_block
from wingjournal.vision.aruco import DEFAULT_DICT, detect_markers
from wingjournal.vision.boundary import best_roles
from wingjournal.vision.hypothesis import ScoringWeights, select_boundary
from wingjournal.vision.literal_box import detect_literal_assets, mask_literals
from wingjournal.vision.orientation import resolve_orientation
from wingjournal.vision.preprocess import preprocess
from wingjournal.vision.rectify import rectify


@dataclass
class IngestResult:
    name: str
    capture: Capture
    raw_image: np.ndarray
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
    recognizer: str = "auto",
    parse_body: bool = False,
) -> IngestResult:
    from wingjournal.recognition.text import get_recognizer

    rec = get_recognizer(recognizer)
    pre = preprocess(image)
    markers = detect_markers(pre.gray, dict_name)

    boundary, hypotheses, squares = select_boundary(pre, markers, weights=weights)

    quad = np.asarray(boundary.polygon, dtype=np.float32)
    provisional, provisional_h = rectify(image, quad)
    orientation = resolve_orientation(pre, markers, boundary.polygon, rectified=provisional)

    if orientation.degrees:
        normalized, homography = rectify(image, quad, rotate_degrees=orientation.degrees)
    else:
        normalized, homography = provisional, provisional_h

    # literal / image regions (spec §16) are detected and masked *before* the
    # detailed recognition stages so their contents never become elements (§36)
    literals = detect_literal_assets(normalized)
    for_parsing = mask_literals(normalized, literals) if literals else normalized

    metadata_block = detect_metadata_block(for_parsing)
    page_metadata = None
    if metadata_block is not None and rec.name != "none":
        from wingjournal.recognition.metadata import read_metadata_block

        reading = read_metadata_block(for_parsing, metadata_block, rec)
        page_metadata = dataclasses.asdict(reading.metadata)
        page_metadata["_confidence"] = reading.confidence

    elements: list[dict] = []
    if parse_body and rec.name != "none":
        from wingjournal.recognition.page_text import recognize_lines
        from wingjournal.recognition.parse import parse_lines

        block_bottom = 0.0
        if metadata_block is not None:
            mb = metadata_block.bbox
            block_bottom = (mb[1] + mb[3]) / for_parsing.shape[0]
        lines = recognize_lines(for_parsing, rec, skip_top=block_bottom)
        elements = [e.to_dict() for e in parse_lines(lines)]

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
        metadata_block=dataclasses.asdict(metadata_block) if metadata_block else None,
        page_metadata=page_metadata,
        text_backend=rec.name,
        literal_assets=[dataclasses.asdict(a) for a in literals],
        detected_elements=elements,
        page_hypotheses=hypotheses,
    )
    capture.notes.append(
        f"{len(markers)} ArUco marker(s), {len(squares)} square candidate(s); "
        f"boundary via {boundary.method} (score {boundary.confidence:.2f}); "
        f"orientation {orientation.degrees} deg via {orientation.method}"
        + (" (flip ambiguous)" if orientation.flip_ambiguous else "")
    )
    if metadata_block is not None:
        capture.notes.append(
            f"metadata block: {len(metadata_block.row1_cells)}+"
            f"{len(metadata_block.row2_cells)} cells (conf {metadata_block.confidence:.2f})"
        )
    if literals:
        capture.notes.append(
            f"{len(literals)} literal image region(s) detected and masked (spec §16)"
        )
    if elements:
        by_kind: dict[str, int] = {}
        for e in elements:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        capture.notes.append(
            "parsed elements: " + ", ".join(f"{v} {k}" for k, v in sorted(by_kind.items()))
        )
    if not markers and not squares:
        capture.notes.append(
            "no fiducial evidence - boundary is a geometric guess (spec sections 27-28)"
        )
    return IngestResult(
        name=name,
        capture=capture,
        raw_image=image,
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
    store=None,
    recognizer: str = "auto",
    parse_body: bool = False,
) -> list[IngestResult]:
    src: CaptureSource = source_for(path, recursive=recursive)
    out_dir = Path(out_dir)
    (out_dir / "normalized").mkdir(parents=True, exist_ok=True)
    (out_dir / "captures").mkdir(parents=True, exist_ok=True)

    results: list[IngestResult] = []
    for name, image in src:
        result = ingest_image(
            name, image, dict_name=dict_name, source_type=src.source_type,
            raw_path=str(path), weights=weights, recognizer=recognizer,
            parse_body=parse_body,
        )
        norm_path = out_dir / "normalized" / f"{name}.png"
        cv2.imwrite(str(norm_path), result.normalized_image)
        result.capture.normalized_image_path = str(norm_path)

        if store is not None:
            from wingjournal.storage import persist_ingest

            ok, buf = cv2.imencode(".png", result.raw_image)
            md = result.capture.page_metadata or {}
            persist_ingest(
                store, result.capture, result.normalized_image,
                buf.tobytes() if ok else b"",
                page_id_explicit=md.get("page_id"),
                topic_tags=md.get("topic_tags") or None,
                document_id_explicit=md.get("document_id"),
            )

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
