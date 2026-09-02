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


def _sheet_with_metadata() -> np.ndarray:
    from wingjournal.recognition.metadata_block import detect_metadata_block
    from wingjournal.templates.geometry import compute_layout
    from wingjournal.templates.writing_sheet import render_writing_sheet

    layout = compute_layout(dpi=200)
    sheet = render_writing_sheet(layout)
    block = detect_metadata_block(sheet)
    assert block is not None
    fills = {0: "Research", 1: "P017", 2: "AI"}
    for i, cell in enumerate(block.row1_cells):
        if i not in fills:
            continue
        x, y, w, h = (int(v) for v in cell)
        cv2.putText(sheet, "#" + fills[i], (x + 12, y + int(h * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return sheet


def test_ingest_reads_metadata_cells(tmp_path):
    from wingjournal.pipeline import ingest_image

    result = ingest_image("s", _sheet_with_metadata(), recognizer="tesseract")
    md = result.capture.page_metadata
    assert md is not None
    assert result.capture.text_backend == "tesseract"
    assert md["page_id"] == "P017"
    assert md["document_id"] == "Research"


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
        page = store.find_page(page_id_explicit="P017")
        assert page is not None
        assert len(page.capture_uuids) == 2  # both captures -> one page
