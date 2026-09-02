import numpy as np
import pytest

from wingjournal.recognition.text import (
    NullRecognizer,
    RecognizedRegion,
    get_recognizer,
)
from wingjournal.recognition.text.segment import segment_lines, segment_words


def test_null_recognizer_is_always_available_and_unreadable():
    rec = NullRecognizer()
    assert rec.available()
    region = rec.recognize(np.zeros((10, 10), np.uint8))
    assert not region.recognized
    assert region.backend == "none"


def test_get_recognizer_none_and_auto_fallback():
    assert get_recognizer("none").name == "none"
    # auto never raises; falls back to null when tesseract is absent
    assert get_recognizer("auto").name in {"none", "tesseract"}


def test_get_recognizer_tesseract_raises_when_missing():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        pytest.skip("tesseract is installed")
    except Exception:
        pass
    with pytest.raises(RuntimeError):
        get_recognizer("tesseract")


def test_unreadable_region_placeholder():
    r = RecognizedRegion.unreadable("x")
    assert r.text and not r.recognized


def _text_image(lines: list[str]) -> np.ndarray:
    import cv2

    img = np.full((60 * len(lines) + 40, 640), 255, np.uint8)
    y = 50
    for ln in lines:
        cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
        y += 60
    return img


def test_segment_lines_and_words():
    img = _text_image(["hello world", "second line here"])
    lines = segment_lines(img)
    assert len(lines) == 2
    assert lines[0][1] < lines[1][1]  # top-to-bottom

    words = segment_words(img, lines[0])
    assert len(words) == 2
