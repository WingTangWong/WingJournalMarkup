"""Capture sources - abstractions over where an image comes from.

The processing pipeline must not depend on the source (spec section 4).
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class CaptureSource(abc.ABC):
    """Yields ``(name, image)`` pairs, where ``image`` is a BGR ``np.ndarray``."""

    source_type: str = "unknown"

    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        raise NotImplementedError


def _imread(path: Path) -> np.ndarray:
    # cv2.imread silently returns None on failure and mangles non-ASCII paths.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


class FileSource(CaptureSource):
    source_type = "file"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        yield self.path.stem, _imread(self.path)


class DirectorySource(CaptureSource):
    source_type = "directory"

    def __init__(self, path: str | Path, recursive: bool = False) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        self.recursive = recursive

    def _files(self) -> list[Path]:
        it = self.path.rglob("*") if self.recursive else self.path.glob("*")
        return sorted(p for p in it if p.suffix.lower() in _IMAGE_SUFFIXES)

    def _name(self, p: Path) -> str:
        # relative path minus suffix, flattened - so a/x.jpg and b/x.png do not
        # collide on "x" in recursive mode. Non-recursive stays just the stem.
        return "__".join(p.relative_to(self.path).with_suffix("").parts)

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        for p in self._files():
            yield self._name(p), _imread(p)


def source_for(path: str | Path, recursive: bool = False) -> CaptureSource:
    """Pick a source based on whether ``path`` is a file or a directory."""

    p = Path(path)
    if p.is_dir():
        return DirectorySource(p, recursive=recursive)
    return FileSource(p)
