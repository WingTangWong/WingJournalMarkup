"""Tesseract backend (local, offline). Optional: needs ``pytesseract`` and the
system ``tesseract-ocr`` binary. Strong on print / block letters, weak on
cursive - which is fine as a first pass; a better handwriting engine can be
dropped in behind the same interface later.
"""

from __future__ import annotations

import cv2
import numpy as np

from wingjournal.recognition.text import (
    RecognizedRegion,
    RecognizedWord,
    TextRecognizer,
)


class TesseractRecognizer(TextRecognizer):
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6) -> None:
        self.lang = lang
        self.psm = psm  # 6 = assume a uniform block of text
        self._checked: bool | None = None

    def available(self) -> bool:
        if self._checked is not None:
            return self._checked
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._checked = True
        except Exception:
            self._checked = False
        return self._checked

    def recognize(self, image: np.ndarray) -> RecognizedRegion:
        if not self.available():
            return RecognizedRegion.unreadable(self.name)

        import pytesseract

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        config = f"--psm {self.psm}"
        try:
            data = pytesseract.image_to_data(
                gray, lang=self.lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            return RecognizedRegion.unreadable(self.name)

        words: list[RecognizedWord] = []
        for text, conf, x, y, w, h in zip(
            data["text"], data["conf"], data["left"], data["top"],
            data["width"], data["height"], strict=True,
        ):
            text = text.strip()
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if not text or c < 0:
                continue
            words.append(RecognizedWord(
                text=text,
                bbox=[float(x - 12), float(y - 12), float(w), float(h)],
                confidence=round(c / 100.0, 3),
            ))

        joined = " ".join(w.text for w in words)
        return RecognizedRegion(
            text=joined, words=words, backend=self.name, recognized=bool(words)
        )
