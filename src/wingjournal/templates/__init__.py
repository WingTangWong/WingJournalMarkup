"""Printable WJM templates.

- ``writing_sheet``: a blank page to write on - four corner ArUco markers and an
  empty two-row metadata block (spec section 11).
- ``legend``: a one-page cheat-sheet of the WJM hand markup.

Both render to a raster page (so the detector sees exactly what is printed) and
are written out as PDF via Pillow.
"""

from wingjournal.templates.geometry import PAPERS_MM, SheetLayout, compute_layout
from wingjournal.templates.legend import build_legend_pdf, render_legend_image
from wingjournal.templates.writing_sheet import (
    build_writing_sheet,
    build_writing_sheet_pdf,
    render_writing_sheet,
)

__all__ = [
    "PAPERS_MM",
    "SheetLayout",
    "compute_layout",
    "build_legend_pdf",
    "render_legend_image",
    "build_writing_sheet",
    "build_writing_sheet_pdf",
    "render_writing_sheet",
]
