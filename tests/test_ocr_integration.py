"""End-to-end OCR path. Skipped unless the tesseract binary is installed
(CI installs it); the rest of the suite exercises the plumbing with fakes.
"""

import cv2
import numpy as np
import pytest

pytest.importorskip("pytesseract")


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _tesseract_available(), reason="tesseract-ocr binary not installed"
)


def _sheet_with_metadata(dpi: int = 300) -> np.ndarray:
    from wingjournal.templates.geometry import compute_layout
    from wingjournal.templates.writing_sheet import render_writing_sheet

    # 300 DPI ≈ a phone photo of a Letter page that fills the frame, so ingest
    # takes its INTER_AREA downscale path (the real-world one) rather than a
    # near-1:1 warp that neither matches a capture nor Tesseract's sweet spot.
    layout = compute_layout(dpi=dpi)
    sheet = render_writing_sheet(layout)
    # OCR-unambiguous, all-caps values (no 0/O or 1/l to trip Tesseract on a
    # small cell); the point is the pipeline + identity, not glyph accuracy
    fills = {"document_id": "RESEARCH", "page_id": "PAGE", "topic_tags": "AI"}
    fs = dpi / 220.0
    for field, value in fills.items():
        x, y, w, h = (int(v) for v in layout.field_box[field])
        cv2.putText(sheet, "#" + value, (x + int(h * 0.2), y + int(h * 0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(2, round(2 * fs)),
                    cv2.LINE_AA)
    return sheet


def test_ingest_reads_metadata_cells(tmp_path):
    from wingjournal.pipeline import ingest_image

    result = ingest_image("s", _sheet_with_metadata(), recognizer="tesseract")
    md = result.capture.page_metadata
    assert md is not None
    assert result.capture.text_backend == "tesseract"
    assert md["page_id"] == "PAGE"
    assert md["document_id"] == "RESEARCH"


def test_ingest_store_uses_metadata_for_identity(tmp_path):
    from wingjournal.pipeline import ingest_path
    from wingjournal.storage import Store

    img = tmp_path / "a.png"
    cv2.imwrite(str(img), _sheet_with_metadata())
    b = tmp_path / "b.png"
    cv2.imwrite(str(b), _sheet_with_metadata())

    with Store(tmp_path / "store") as store:
        ingest_path(img, tmp_path / "out", store=store, recognizer="tesseract")
        ingest_path(b, tmp_path / "out", store=store, recognizer="tesseract")
        page = store.find_page(page_id_explicit="PAGE")
        assert page is not None
        assert len(page.capture_uuids) == 2  # both captures -> one page
