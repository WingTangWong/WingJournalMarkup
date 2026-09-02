"""Parse recognized text lines into typed WJM elements (spec §18-22).

Text-level only: bullets, tag lines, temporal tags, references, contact blocks.
Box geometry (`DiagramNode`) and diagram edges are M5/M6 and not here yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from wingjournal.recognition.page_text import TextLine
from wingjournal.recognition.tags import parse_reference, parse_tags

# leading glyph -> bullet-journal state (spec §18); a few pen-friendly aliases
BULLET_STATES: dict[str, str] = {
    "•": "open", "*": "open", "·": "open",
    "x": "completed", "×": "completed", "X": "completed",
    ">": "migrated", "<": "scheduled",
    "-": "note", "–": "note", "—": "note",
    "o": "event", "O": "event", "○": "event", "◦": "event",
    "!": "important", "?": "question",
}

_TEMPORAL_RE = re.compile(
    r"\[\s*(DUE|EVENT|RANGE)\s*:\s*(?P<a>.+?)\s*(?:->\s*(?P<b>.+?)\s*)?\]",
    re.IGNORECASE,
)
_CONTACT_START_RE = re.compile(r"^\+?\s*CONTACT\b", re.IGNORECASE)
_CONTACT_END_RE = re.compile(r"^\+[-\s]*\+?\s*$")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-.\s()]{6,}\d)")


@dataclass
class Element:
    kind: str  # text | bullet | tags | temporal | reference | contact | heading
    text: str
    bbox: list[float]
    confidence: float = 0.0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "text": self.text, "bbox": self.bbox,
            "confidence": self.confidence, "data": self.data,
        }


def _temporal_elements(line: TextLine) -> list[Element]:
    out = []
    for m in _TEMPORAL_RE.finditer(line.text):
        kind = m.group(1).lower()
        data = {"type": kind, "start": m.group("a").strip(), "match": m.group(0)}
        if m.group("b"):
            data["end"] = m.group("b").strip()
        out.append(Element("temporal", line.text, line.bbox, line.confidence, data))
    return out


def _bullet(line: TextLine) -> Element | None:
    s = line.text.lstrip()
    if len(s) < 2 or not s[1].isspace():
        return None
    state = BULLET_STATES.get(s[0])
    if state is None:
        return None
    body = s[2:].strip()
    return Element(
        "bullet", line.text, line.bbox, line.confidence,
        {"state": state, "glyph": s[0], "item": body, "tags": parse_tags(body)},
    )


def parse_lines(lines: list[TextLine]) -> list[Element]:
    elements: list[Element] = []
    contact: list[TextLine] | None = None

    for line in lines:
        text = line.text.strip()
        if contact is not None:
            if _CONTACT_END_RE.match(text) or not text:
                elements.append(_finish_contact(contact))
                contact = None
                continue
            contact.append(line)
            continue
        if not text:
            continue

        if _CONTACT_START_RE.match(text):
            contact = []
            continue

        temporal = _temporal_elements(line)
        elements.extend(temporal)

        ref = parse_reference(text)
        if ref is not None:
            elements.append(Element(
                "reference", text, line.bbox, line.confidence,
                {"anchor": ref.anchor, "document": ref.document, "page": ref.page},
            ))
            continue

        bullet = _bullet(line)
        if bullet is not None:
            elements.append(bullet)
            continue

        tags = parse_tags(text)
        stripped = re.sub(r"#(?:\[[^\]]+\]|\S+)", "", text).strip()
        if tags and not stripped:
            elements.append(Element("tags", text, line.bbox, line.confidence, {"tags": tags}))
            continue

        if temporal:
            continue  # the temporal element(s) already captured this line

        kind = "heading" if len(text) <= 40 and text == text.title() else "text"
        elements.append(Element(kind, text, line.bbox, line.confidence,
                                {"tags": tags} if tags else {}))

    if contact is not None:
        elements.append(_finish_contact(contact))
    return elements


def _finish_contact(lines: list[TextLine]) -> Element:
    joined = "\n".join(line.text.strip() for line in lines if line.text.strip())
    email = _EMAIL_RE.search(joined)
    phone = _PHONE_RE.search(joined)
    remaining = joined
    for hit in (email, phone):
        if hit:
            remaining = remaining.replace(hit.group(0), "")
    parts = [p.strip() for p in remaining.splitlines() if p.strip()]
    data = {
        "name": parts[0] if parts else None,
        "email": email.group(0) if email else None,
        "phone": phone.group(0).strip() if phone else None,
        "organization": parts[1] if len(parts) > 1 else None,
    }
    bbox = lines[0].bbox if lines else [0, 0, 0, 0]
    conf = round(sum(line.confidence for line in lines) / len(lines), 3) if lines else 0.0
    return Element("contact", joined, bbox, conf, data)
