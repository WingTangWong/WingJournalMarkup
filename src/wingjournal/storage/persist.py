"""Bridge from an ingest result to the store."""

from __future__ import annotations

import cv2

from wingjournal.models import Capture
from wingjournal.storage.identity import IdentityResult, resolve_identity
from wingjournal.storage.store import Store


def persist_ingest(
    store: Store,
    capture: Capture,
    normalized_image,
    raw_bytes: bytes,
    *,
    page_id_machine: str | None = None,
    page_id_explicit: str | None = None,
    topic_tags: list[str] | None = None,
    document_id_explicit: str | None = None,
) -> tuple[Capture, IdentityResult]:
    """Store the raw + normalized blobs, resolve the page, insert the capture.

    Identity inputs are optional - the pipeline cannot supply page ids until OCR
    (M4), so today every capture resolves to a fresh page.
    """

    capture.raw_blob = store.put_blob(raw_bytes)
    ok, buf = cv2.imencode(".png", normalized_image)
    if not ok:
        raise RuntimeError("failed to encode normalized image")
    capture.normalized_blob = store.put_blob(buf.tobytes())

    # store each literal region's interior as its own blob (spec §16)
    for asset in capture.literal_assets:
        x, y, w, h = (int(round(v)) for v in asset["bbox"])
        pad = max(2, int(min(w, h) * 0.1))
        crop = normalized_image[y + pad : y + h - pad, x + pad : x + w - pad]
        if crop.size:
            ok, cbuf = cv2.imencode(".png", crop)
            if ok:
                asset["asset_blob"] = store.put_blob(cbuf.tobytes())

    ident = resolve_identity(
        store,
        page_id_machine=page_id_machine,
        page_id_explicit=page_id_explicit,
        topic_tags=topic_tags,
        document_id_explicit=document_id_explicit,
        capture_uuid=capture.uuid,
    )
    capture.page_uuid = ident.page.uuid
    store.add_capture(capture)
    return capture, ident
