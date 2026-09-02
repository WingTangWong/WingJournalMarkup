"""Text / handwriting recognition (spec §48, roadmap M4).

A ``TextRecognizer`` turns a normalized-page region into text with per-word
boxes and confidence. Recognition is always optional: when no engine is
available every region comes back as an *unrecognized* placeholder, and the
rest of the pipeline carries on.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

UNRECOGNIZED = "�"  # placeholder text for a region we could not read


@dataclass
class RecognizedWord:
    text: str
    bbox: list[float]  # [x, y, w, h] in the region's own pixel coords
    confidence: float


@dataclass
class RecognizedRegion:
    text: str = ""
    words: list[RecognizedWord] = field(default_factory=list)
    backend: str = "none"
    recognized: bool = False

    @classmethod
    def unreadable(cls, backend: str) -> RecognizedRegion:
        return cls(text=UNRECOGNIZED, backend=backend, recognized=False)


class TextRecognizer(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def available(self) -> bool:
        ...

    @abc.abstractmethod
    def recognize(self, image: np.ndarray) -> RecognizedRegion:
        ...


class NullRecognizer(TextRecognizer):
    """Always 'unrecognized' - the guaranteed fallback."""

    name = "none"

    def available(self) -> bool:
        return True

    def recognize(self, image: np.ndarray) -> RecognizedRegion:
        return RecognizedRegion.unreadable(self.name)


def get_recognizer(prefer: str = "auto") -> TextRecognizer:
    """``auto`` uses Tesseract when it is importable *and* its binary is on
    PATH, else the null recognizer. ``tesseract`` / ``none`` force one."""

    if prefer in ("auto", "tesseract"):
        try:
            from wingjournal.recognition.text.tesseract import TesseractRecognizer

            rec = TesseractRecognizer()
            if rec.available():
                return rec
        except Exception:
            pass
        if prefer == "tesseract":
            raise RuntimeError("tesseract recognizer requested but not available")
    return NullRecognizer()


__all__ = [
    "UNRECOGNIZED",
    "RecognizedWord",
    "RecognizedRegion",
    "TextRecognizer",
    "NullRecognizer",
    "get_recognizer",
]
