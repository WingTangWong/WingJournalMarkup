"""The WJM hand-markup grammar: tags, metadata cells, and references.

Pure text parsing (spec §10, §11, §19) - no image work. M4 feeds OCR output
through these; the parsers are exercised now with literal strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# #term  or  #[term with spaces]
_TAG_RE = re.compile(r"#(?:\[(?P<bracketed>[^\]\n]+)\]|(?P<word>[^\s#\[\]]+))")

# a reference enclosure: "-> [ ... ]" (anywhere) or "REF: ..." / "-> ..." (rest of line)
_REF_BRACKET_RE = re.compile(r"(?:->|REF:)\s*\[\s*(?P<body>[^\]\n]+?)\s*\]", re.IGNORECASE)
_REF_BARE_RE = re.compile(r"(?:->|REF:)\s*(?P<body>[^\]\n]+?)\s*$", re.IGNORECASE)


def canonical(name: str) -> str:
    """Trim and collapse internal whitespace."""

    return re.sub(r"\s+", " ", name).strip()


def parse_tag(token: str) -> str | None:
    """A single token -> canonical tag string, or ``None`` if it is not a tag.

    ``#auth`` -> ``auth``; ``#[Data Science]`` -> ``Data Science``.
    """

    m = _TAG_RE.fullmatch(token.strip())
    if not m:
        return None
    return canonical(m.group("bracketed") or m.group("word"))


def parse_tags(text: str) -> list[str]:
    """Every tag in ``text``, in order, canonicalized (duplicates kept)."""

    return [
        canonical(m.group("bracketed") or m.group("word"))
        for m in _TAG_RE.finditer(text or "")
    ]


def first_tag(text: str) -> str | None:
    """The first tag in ``text`` - for single-value cells (doc id, page id, L/A/B/R)."""

    tags = parse_tags(text)
    return tags[0] if tags else None


@dataclass
class PageMetadata:
    """Parsed page-metadata block (spec §11)."""

    document_id: str | None = None
    page_id: str | None = None
    topic_tags: list[str] = field(default_factory=list)
    left: str | None = None
    above: str | None = None
    below: str | None = None
    right: str | None = None


def parse_metadata_cells(
    row1: list[str | None], row2: list[str | None]
) -> PageMetadata:
    """``row1`` = [document, page, topics]; ``row2`` = [left, above, below, right].

    Missing / blank cells are fine (spec §11); pass ``None`` or ``""``.
    """

    r1 = list(row1) + [None] * (3 - len(row1))
    r2 = list(row2) + [None] * (4 - len(row2))
    return PageMetadata(
        document_id=first_tag(r1[0] or ""),
        page_id=first_tag(r1[1] or ""),
        topic_tags=parse_tags(r1[2] or ""),
        left=first_tag(r2[0] or ""),
        above=first_tag(r2[1] or ""),
        below=first_tag(r2[2] or ""),
        right=first_tag(r2[3] or ""),
    )


@dataclass
class Reference:
    """A resolved-as-far-as-possible address link (spec §19)."""

    anchor: str
    document: str | None = None
    page: str | None = None
    raw: str = ""


def split_qualified(name: str) -> tuple[str | None, str | None, str]:
    """``document : page : anchor`` -> ``(document, page, anchor)``.

    Fewer components fill from the right: ``P017:AUTH`` -> ``(None, "P017", "AUTH")``,
    ``AUTH`` -> ``(None, None, "AUTH")``. A leading ``#`` and ``#[...]`` quoting
    on any component are accepted.
    """

    parts = [canonical(p) for p in name.split(":")]
    parts = [parse_tag(p) or p for p in parts if p != ""]
    if not parts:
        return None, None, ""
    anchor = parts[-1]
    page = parts[-2] if len(parts) >= 2 else None
    document = parts[-3] if len(parts) >= 3 else None
    return document, page, anchor


def parse_reference(text: str) -> Reference | None:
    """A reference enclosure -> :class:`Reference`, or ``None``.

    Accepts ``-> [#AUTH]``, ``REF: #AUTH``, ``-> [Research:P017:AUTH]``,
    ``-> [#[Auth Service]]``.
    """

    m = _REF_BRACKET_RE.search(text or "") or _REF_BARE_RE.search(text or "")
    if not m:
        return None
    # tolerate a #[...] tag nested in the enclosure: drop stray enclosure chars
    body = m.group("body").strip().strip("[]").strip()
    if ":" in body:
        document, page, anchor = split_qualified(body)
    else:
        anchor = parse_tag(body) or canonical(body.lstrip("#["))
        document = page = None
    if not anchor:
        return None
    return Reference(anchor=anchor, document=document, page=page, raw=text.strip())
