"""Presentation helpers: map parsed elements to CSS classes / labels."""

from __future__ import annotations

KIND_CLASS = {
    "heading": "el-heading",
    "text": "el-text",
    "bullet": "el-bullet",
    "tags": "el-tags",
    "temporal": "el-temporal",
    "reference": "el-reference",
    "contact": "el-contact",
}

BULLET_GLYPH = {
    "open": "•", "completed": "✓", "migrated": "→", "scheduled": "◁",
    "note": "–", "event": "○", "important": "!", "question": "?",
}


def element_view(el: dict) -> dict:
    kind = el["kind"]
    v = {
        "kind": kind,
        "cls": KIND_CLASS.get(kind, "el-text"),
        "text": el.get("text", ""),
        "confidence": el.get("confidence", 0.0),
        "data": el.get("data", {}),
        "bbox": el.get("bbox"),
    }
    d = el.get("data", {})
    if kind == "bullet":
        v["glyph"] = BULLET_GLYPH.get(d.get("state"), "•")
        v["label"] = d.get("state", "")
        v["text"] = d.get("item", v["text"])
        v["tags"] = d.get("tags", [])
    elif kind == "tags":
        v["tags"] = d.get("tags", [])
    elif kind == "temporal":
        parts = [d.get("type", "").upper(), d.get("start", "")]
        if d.get("end"):
            parts.append("→ " + d["end"])
        v["label"] = " ".join(parts)
    elif kind == "reference":
        ref = ":".join(x for x in (d.get("document"), d.get("page"), d.get("anchor")) if x)
        v["label"] = ref
    elif kind == "contact":
        v["contact"] = d
    return v


def summarize_elements(elements: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in elements:
        out[e["kind"]] = out.get(e["kind"], 0) + 1
    return out
