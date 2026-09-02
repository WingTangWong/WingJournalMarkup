"""Full-page line recognition: segment the normalized page into text lines and
run each through a :class:`TextRecognizer`. Literal regions must already be
masked by the caller (spec §36).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wingjournal.recognition.text import TextRecognizer
from wingjournal.recognition.text.segment import segment_lines


@dataclass
class TextLine:
    text: str
    bbox: list[float]  # [x, y, w, h] in normalized-page coords
    confidence: float
    recognized: bool


def recognize_lines(
    image: np.ndarray, recognizer: TextRecognizer, skip_top: float = 0.0
) -> list[TextLine]:
    """Recognize every text line on ``image``.

    ``skip_top`` (0..1) drops lines whose top is above that fraction of the page
    - handy for excluding the metadata block, which is read separately.
    """

    h = image.shape[0]
    cutoff = skip_top * h
    out: list[TextLine] = []
    for x, y, w, lh in segment_lines(image):
        if y < cutoff:
            continue
        region = recognizer.recognize(image[y : y + lh, x : x + w])
        conf = (
            sum(word.confidence for word in region.words) / len(region.words)
            if region.words
            else 0.0
        )
        out.append(TextLine(
            text=region.text if region.recognized else "",
            bbox=[float(x), float(y), float(w), float(lh)],
            confidence=round(conf, 3),
            recognized=region.recognized,
        ))
    return out
