"""Upload -> archive -> ingest -> persist -> diff."""

from __future__ import annotations

import uuid as _uuid

import cv2
import numpy as np

from wingjournal.models import PageRelationship
from wingjournal.pipeline import ingest_image
from wingjournal.storage.persist import persist_ingest
from wjm_demo.diff import diff_captures, is_empty
from wjm_demo.store import DemoStore

_RECIPROCAL = {"LEFT": "RIGHT", "RIGHT": "LEFT", "ABOVE": "BELOW", "BELOW": "ABOVE"}


def ingest_upload(store: DemoStore, original_name: str, data: bytes) -> dict:
    """Returns ``{capture, page, scan_uuid, diff}``."""

    with store.write_lock:
        return _ingest_upload(store, original_name, data)


def _ingest_upload(store: DemoStore, original_name: str, data: bytes) -> dict:
    scan_uuid = str(_uuid.uuid4())
    store.archive_scan(scan_uuid, original_name, data)

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the uploaded image")

    result = ingest_image(scan_uuid, image, parse_body=True)
    cap = result.capture

    md = cap.page_metadata or {}
    prev_caps: list[dict] = []
    # persist_ingest resolves identity; grab the prior captures first for the diff
    if md.get("page_id"):
        existing = store.wjm.find_page(page_id_explicit=md["page_id"])
        if existing is not None:
            prev_caps = store.wjm.captures_for_page(existing.uuid)

    raw_bytes = cv2.imencode(".png", image)[1].tobytes()
    persist_ingest(
        store.wjm, cap, result.normalized_image, raw_bytes,
        page_id_explicit=md.get("page_id"),
        topic_tags=md.get("topic_tags") or None,
        document_id_explicit=md.get("document_id"),
    )
    store.link_scan(scan_uuid, cap.uuid, cap.page_uuid)
    page = store.wjm.get_page(cap.page_uuid)
    _infer_relationships(store, page, cap.uuid)

    diff = None
    if prev_caps:
        import dataclasses

        d = diff_captures(prev_caps[-1], dataclasses.asdict(cap))
        if not is_empty(d):
            diff = d
        store.save_diff(cap.uuid, prev_caps[-1]["uuid"], cap.page_uuid, d)

    return {"capture": cap, "page": page, "scan_uuid": scan_uuid, "diff": diff,
            "normalized_image": result.normalized_image}


def _infer_relationships(store: DemoStore, page, capture_uuid: str) -> None:
    """Spatial linkage from the page's L/A/B/R metadata refs (spec §12-13)."""

    for attr, relation in (("left", "LEFT"), ("above", "ABOVE"),
                           ("below", "BELOW"), ("right", "RIGHT")):
        ref = getattr(page, attr)
        if not ref:
            continue
        target = store.wjm.find_page(page_id_explicit=ref)
        if target is None or target.uuid == page.uuid:
            continue
        store.wjm.add_relationship(PageRelationship(
            source_page=page.uuid, target_page=target.uuid, relation=relation,
            explicitly_declared=True, source_capture=capture_uuid,
        ))
        store.wjm.add_relationship(PageRelationship(
            source_page=target.uuid, target_page=page.uuid,
            relation=_RECIPROCAL[relation], explicitly_declared=False,
            source_capture=capture_uuid, confidence=0.8,
        ))
