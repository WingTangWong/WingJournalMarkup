"""Read a detected metadata block's cells into a :class:`PageMetadata`.

Ties together M3's block geometry, M4's text recognition, and the tag grammar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wingjournal.recognition.metadata_block import MetadataBlock
from wingjournal.recognition.tags import PageMetadata, parse_metadata_cells
from wingjournal.recognition.text import RecognizedRegion, TextRecognizer


@dataclass
class MetadataReading:
    metadata: PageMetadata
    row1_text: list[str]
    row2_text: list[str]
    confidence: float  # mean word confidence over the cells that had text


def _crop(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    x, y, w, h = (int(round(v)) for v in bbox)
    H, W = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), np.uint8)
    return image[y0:y1, x0:x1]


def _pad(cells: list, n: int) -> list:
    return (list(cells) + [None] * n)[:n]


def read_metadata_block(
    normalized_image: np.ndarray,
    block: MetadataBlock,
    recognizer: TextRecognizer,
) -> MetadataReading:
    confs: list[float] = []

    def cell_text(bbox) -> str:
        if bbox is None:
            return ""
        region: RecognizedRegion = recognizer.recognize(_crop(normalized_image, bbox))
        if region.recognized:
            confs.extend(w.confidence for w in region.words)
            return region.text
        return ""

    row1 = [cell_text(b) for b in _pad(block.row1_cells, 3)]
    row2 = [cell_text(b) for b in _pad(block.row2_cells, 4)]
    return MetadataReading(
        metadata=parse_metadata_cells(row1, row2),
        row1_text=row1,
        row2_text=row2,
        confidence=round(sum(confs) / len(confs), 3) if confs else 0.0,
    )
