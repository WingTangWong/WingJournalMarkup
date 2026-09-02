"""Shared rendering helpers for the templates: fonts and PDF output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    "/Library/Fonts",
    "/usr/share/fonts/TTF",
)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """A scalable font that works with or without system DejaVu installed."""

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in _FONT_DIRS:
        path = Path(d) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    # Pillow >= 10.1 ships a scalable fallback.
    return ImageFont.load_default(size=size)


def to_pil(image: np.ndarray) -> Image.Image:
    """BGR or grayscale ndarray -> RGB PIL image."""

    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image[:, :, ::-1])  # BGR -> RGB


def save_pdf(pages: list[Image.Image | np.ndarray], out: str | Path, dpi: int) -> Path:
    if not pages:
        raise ValueError("no pages to write")
    imgs = [to_pil(p) if isinstance(p, np.ndarray) else p.convert("RGB") for p in pages]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        out,
        "PDF",
        resolution=float(dpi),
        save_all=True,
        append_images=imgs[1:],
    )
    return out
