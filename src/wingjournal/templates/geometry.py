"""Page-layout math for the printable templates.

Everything is specified in millimetres and converted to pixels at a chosen DPI,
so a printed sheet has physically predictable marker sizes (which is what the
detector's confidence tiers assume).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wingjournal.vision.aruco import (
    MARKER_ROLE_IDS,
    METADATA_FIELD_ANCHOR_MM,
    METADATA_FIELD_IDS,
)

# width x height in millimetres, portrait
PAPERS_MM: dict[str, tuple[float, float]] = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
    "legal": (215.9, 355.6),
}

__all__ = ["PAPERS_MM", "MARKER_ROLE_IDS", "SheetLayout", "compute_layout", "mm_to_px"]

# Row 1 / Row 2 fields for the metadata block (spec section 11). The keys line
# up with METADATA_FIELD_IDS; the values are the printed human captions.
METADATA_ROW1_FIELDS = ("document_id", "page_id", "topic_tags")
METADATA_ROW2_FIELDS = ("left", "above", "below", "right")
METADATA_ROW1 = ("DOCUMENT ID", "PAGE ID", "TOPIC TAGS")
METADATA_ROW2 = ("LEFT", "ABOVE", "BELOW", "RIGHT")

# 0.25" - the widest non-printable border a consumer inkjet/laser typically
# imposes. Corner markers sit their outer edge on this line, i.e. as close to
# the paper corner as a normal printer can render them.
PRINT_MARGIN_MM = 25.4 / 4  # 6.35

# Metadata-block row height: 0.45" per row, two rows (spec section 11). Bigger
# than the old 1/4" so a field's ArUco anchor, its caption and a usable writing
# box all fit; the corner markers grow to match so both rows still nest inside
# the top-marker height (see _DEFAULT_MARKER_MM).
METADATA_ROW_MM = 25.4 * 0.45  # 11.43

# Corner marker side. 24 mm both keeps 2x 0.45" rows inside the marker height and
# reads more reliably from a phone photo than the old 18 mm.
_DEFAULT_MARKER_MM = 24.0

# Concentric-square registration mark at each metadata-block corner (spec §11).
# Big enough to keep its rings resolvable after a phone photo + rectify, small
# enough to straddle a corner without crossing an ArUco quiet zone.
REGISTRATION_MARK_MM = 4.2


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
    # concentric-square registration marks: centre (x, y) at each block corner
    registration_xy: tuple[tuple[int, int], ...] = ()
    registration_px: int = 0
    row_captions: tuple[tuple[str, ...], tuple[str, ...]] = field(
        default=(METADATA_ROW1, METADATA_ROW2)
    )
    # per-field ArUco anchor bitmaps: field name -> (x, y) top-left, all square
    # and ``field_anchor_px`` on a side
    field_anchor_xy: dict[str, tuple[int, int]] = field(default_factory=dict)
    field_anchor_px: int = 0
    # per-field OCR target box: field name -> (x, y, w, h), the zone to the right
    # of that field's anchor
    field_box: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    @property
    def marker_ids(self) -> dict[str, int]:
        return dict(MARKER_ROLE_IDS)

    @property
    def field_ids(self) -> dict[str, int]:
        return dict(METADATA_FIELD_IDS)


def compute_layout(
    paper: str = "letter",
    dpi: int = 300,
    margin_mm: float = PRINT_MARGIN_MM,
    marker_mm: float = _DEFAULT_MARKER_MM,
    quiet_mm: float = 4.0,
    metadata_gap_mm: float = 8.0,
    metadata_row_mm: float = METADATA_ROW_MM,
    registration_mark_mm: float = REGISTRATION_MARK_MM,
    field_anchor_mm: float = METADATA_FIELD_ANCHOR_MM,
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
    # top and bottom edges of the top corner codes. A 2x-quiet side gap keeps
    # the block clear of each marker's quiet zone.
    side_gap = 2 * quiet
    row_h = mm_to_px(metadata_row_mm, dpi)
    meta_h = 2 * row_h
    meta_x = margin + marker + side_gap
    meta_w = W - 2 * meta_x
    meta_y = max(margin, margin + (marker - meta_h) // 2)

    # registration marks straddle the four block corners (centre on the corner)
    reg = mm_to_px(registration_mark_mm, dpi)
    registration_xy = (
        (meta_x, meta_y),
        (meta_x + meta_w, meta_y),
        (meta_x + meta_w, meta_y + meta_h),
        (meta_x, meta_y + meta_h),
    )

    # Per-field layout: [anchor][gap][box]  ... trailing whitespace ...  repeat.
    # Field content is inset from the block edges to clear the corner marks; each
    # row is split into equal slots (3 for row 1, 4 for row 2).
    anchor = mm_to_px(field_anchor_mm, dpi)
    edge = round(reg * 1.45 / 2) + mm_to_px(2.5, dpi)
    a_gap = mm_to_px(1.6, dpi)          # anchor -> its box
    trail = mm_to_px(3.0, dpi)          # box end -> next slot
    cap_h = max(anchor // 3, mm_to_px(2.6, dpi))  # caption strip at the row top
    vpad = mm_to_px(1.0, dpi)
    band_x = meta_x + edge
    band_w = meta_w - 2 * edge

    field_anchor_xy: dict[str, tuple[int, int]] = {}
    field_box: dict[str, tuple[int, int, int, int]] = {}
    for row_i, fields in enumerate((METADATA_ROW1_FIELDS, METADATA_ROW2_FIELDS)):
        n = len(fields)
        slot_w = band_w / n
        row_y = meta_y + row_i * row_h
        anchor_y = row_y + (row_h - anchor) // 2
        box_y = row_y + cap_h
        box_h = max(1, row_h - cap_h - vpad)
        for j, name in enumerate(fields):
            slot_x = band_x + round(j * slot_w)
            slot_end = band_x + round((j + 1) * slot_w)
            field_anchor_xy[name] = (slot_x, anchor_y)
            bx = slot_x + anchor + a_gap
            field_box[name] = (bx, box_y, max(1, slot_end - trail - bx), box_h)

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
        # body starts below whichever reaches lower: the top-marker band or the
        # metadata block
        body_top_px=max(
            margin + marker + mm_to_px(metadata_gap_mm, dpi),
            meta_y + meta_h + mm_to_px(metadata_gap_mm, dpi),
        ),
        margin_px=margin,
        registration_xy=registration_xy,
        registration_px=reg,
        field_anchor_xy=field_anchor_xy,
        field_anchor_px=anchor,
        field_box=field_box,
    )
