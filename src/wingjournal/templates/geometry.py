"""Page-layout math for the printable templates.

Everything is specified in millimetres and converted to pixels at a chosen DPI,
so a printed sheet has physically predictable marker sizes (which is what the
detector's confidence tiers assume).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wingjournal.vision.aruco import MARKER_ROLE_IDS

# width x height in millimetres, portrait
PAPERS_MM: dict[str, tuple[float, float]] = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
    "legal": (215.9, 355.6),
}

__all__ = ["PAPERS_MM", "MARKER_ROLE_IDS", "SheetLayout", "compute_layout", "mm_to_px"]

# Row 1 / Row 2 cell captions for the metadata block (spec section 11).
METADATA_ROW1 = ("DOCUMENT ID", "PAGE ID", "TOPIC TAGS")
METADATA_ROW2 = ("LEFT", "ABOVE", "BELOW", "RIGHT")

# 0.25" - the widest non-printable border a consumer inkjet/laser typically
# imposes. Corner markers sit their outer edge on this line, i.e. as close to
# the paper corner as a normal printer can render them.
PRINT_MARGIN_MM = 25.4 / 4  # 6.35

# Metadata-block row height: 1/4" per row, two rows (spec section 11).
METADATA_ROW_MM = 25.4 / 4  # 6.35


def mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


@dataclass
class SheetLayout:
    paper: str
    dpi: int
    width_px: int
    height_px: int
    marker_px: int
    quiet_px: int
    # top-left (x, y) of each marker bitmap, keyed by role
    marker_xy: dict[str, tuple[int, int]]
    # metadata block outer rectangle: (x, y, w, h)
    metadata_rect: tuple[int, int, int, int]
    metadata_row1_h: int
    body_top_px: int
    margin_px: int
    row_captions: tuple[tuple[str, ...], tuple[str, ...]] = field(
        default=(METADATA_ROW1, METADATA_ROW2)
    )

    @property
    def marker_ids(self) -> dict[str, int]:
        return dict(MARKER_ROLE_IDS)


def compute_layout(
    paper: str = "letter",
    dpi: int = 300,
    margin_mm: float = PRINT_MARGIN_MM,
    marker_mm: float = 18.0,
    quiet_mm: float = 4.0,
    metadata_gap_mm: float = 8.0,
    metadata_row_mm: float = METADATA_ROW_MM,
) -> SheetLayout:
    try:
        w_mm, h_mm = PAPERS_MM[paper]
    except KeyError as exc:
        raise ValueError(f"unknown paper {paper!r}; try {sorted(PAPERS_MM)}") from exc

    W = mm_to_px(w_mm, dpi)
    H = mm_to_px(h_mm, dpi)
    margin = mm_to_px(margin_mm, dpi)
    marker = mm_to_px(marker_mm, dpi)
    quiet = mm_to_px(quiet_mm, dpi)

    right_x = W - margin - marker
    bottom_y = H - margin - marker
    marker_xy = {
        "TOP_LEFT": (margin, margin),
        "TOP_RIGHT": (right_x, margin),
        "BOTTOM_RIGHT": (right_x, bottom_y),
        "BOTTOM_LEFT": (margin, bottom_y),
    }

    # The metadata block sits in the clear span *between* the two top markers,
    # vertically centred within the marker height so both rows stay inside the
    # top and bottom edges of the top corner codes. Rows are metadata_row_mm
    # (1/4"), clamped so 2 rows never exceed the marker height for a small
    # --marker-mm. A 2x-quiet side gap keeps the ruling clear of each marker's
    # quiet zone.
    side_gap = 2 * quiet
    row_h = min(mm_to_px(metadata_row_mm, dpi), marker // 2)
    meta_h = 2 * row_h
    meta_x = margin + marker + side_gap
    meta_w = W - 2 * meta_x
    meta_y = margin + (marker - meta_h) // 2

    return SheetLayout(
        paper=paper,
        dpi=dpi,
        width_px=W,
        height_px=H,
        marker_px=marker,
        quiet_px=quiet,
        marker_xy=marker_xy,
        metadata_rect=(meta_x, meta_y, meta_w, meta_h),
        metadata_row1_h=row_h,
        # body starts below the whole top-marker band
        body_top_px=margin + marker + mm_to_px(metadata_gap_mm, dpi),
        margin_px=margin,
    )
