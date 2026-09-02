"""Reassemble a processed capture into a downloadable PDF: the normalized page
with the detected structure drawn on it, plus a text sheet of the extraction.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from wingjournal.templates._render import load_font
from wjm_demo.store import DemoStore

_COLOURS = {
    "heading": (40, 70, 140), "text": (90, 90, 90), "bullet": (20, 130, 60),
    "tags": (150, 90, 20), "temporal": (170, 40, 120), "reference": (30, 110, 160),
    "contact": (120, 60, 160), "metadata": (200, 130, 0), "literal": (200, 40, 40),
}


def _overlay_page(store: DemoStore, cap: dict) -> Image.Image:
    blob = cap.get("normalized_blob")
    if not blob:
        return Image.new("RGB", (1000, 1400), "white")
    img = Image.open(io.BytesIO(store.wjm.get_blob(blob))).convert("RGB")
    draw = ImageDraw.Draw(img)
    lw = max(2, img.width // 500)

    mb = cap.get("metadata_block")
    if mb:
        x, y, w, h = mb["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=_COLOURS["metadata"], width=lw + 1)
        for cell in mb.get("row1_cells", []) + mb.get("row2_cells", []):
            cx, cy, cw, ch = cell
            draw.rectangle([cx, cy, cx + cw, cy + ch], outline=_COLOURS["metadata"], width=1)

    for lit in cap.get("literal_assets", []):
        x, y, w, h = lit["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=_COLOURS["literal"], width=lw + 1)
        draw.text((x + 4, y + 4), "LITERAL", fill=_COLOURS["literal"], font=load_font(20))

    for el in cap.get("detected_elements", []):
        if not el.get("bbox"):
            continue
        x, y, w, h = el["bbox"]
        colour = _COLOURS.get(el["kind"], _COLOURS["text"])
        draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], outline=colour, width=lw)
        draw.text((x + w + 4, y), el["kind"], fill=colour, font=load_font(18))
    return img


def _text_sheet(store: DemoStore, cap: dict, page) -> Image.Image:
    W, H = 1240, 1750
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    h1 = load_font(40, bold=True)
    h2 = load_font(26, bold=True)
    body = load_font(22)
    mono = load_font(20)
    y = 48

    def line(text: str, font, fill, indent: int = 48, gap: int = 30) -> None:
        nonlocal y
        d.text((indent, y), text, font=font, fill=fill)
        y += gap

    line("Wing Journal Markup - extracted document", h1, (40, 70, 140), gap=66)
    line(f"page {page.uuid}", mono, (120, 120, 120))
    line(f"captured {cap.get('timestamp', '')}", mono, (120, 120, 120), gap=44)

    md = cap.get("page_metadata") or {}
    line("Page metadata", h2, (20, 20, 20), gap=36)
    line(f"document_id: {md.get('document_id') or '-'}", body, (20, 20, 20), 64)
    line(f"page_id: {md.get('page_id') or '-'}", body, (20, 20, 20), 64)
    line(f"topics: {', '.join(md.get('topic_tags') or []) or '-'}", body, (20, 20, 20), 64)
    line(f"neighbours: L={page.left} A={page.above} B={page.below} R={page.right}",
         body, (20, 20, 20), 64, gap=44)

    line("Elements", h2, (20, 20, 20), gap=36)
    elements = cap.get("detected_elements", [])
    for el in elements:
        if y > H - 60:
            line("...", body, (120, 120, 120), 64)
            break
        tag = el["kind"].upper()
        if el["kind"] == "bullet":
            tag = f"BULLET/{el['data'].get('state', '').upper()}"
        line(f"[{tag}] {el.get('text', '')}"[:110], body,
             _COLOURS.get(el["kind"], _COLOURS["text"]), 64)
    if not elements:
        line("(no body text extracted - install tesseract-ocr for OCR)",
             body, (150, 150, 150), 64)
    return img


def build_pdf(store: DemoStore, capture_uuid: str) -> bytes:
    cap = store.wjm.get_capture(capture_uuid)
    if cap is None:
        raise KeyError(capture_uuid)
    page = store.wjm.get_page(cap["page_uuid"]) if cap.get("page_uuid") else None
    pages = [_overlay_page(store, cap)]
    if page is not None:
        pages.append(_text_sheet(store, cap, page))
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    return buf.getvalue()
