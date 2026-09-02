from wingjournal.recognition.page_text import TextLine
from wingjournal.recognition.parse import BULLET_STATES, parse_lines


def _line(text: str, y: int = 0) -> TextLine:
    return TextLine(text=text, bbox=[10, y, 300, 20], confidence=0.9, recognized=True)


def test_bullet_states():
    els = parse_lines([_line("* open task"), _line("x done"), _line("> migrated"),
                       _line("? research question")])
    assert [e.kind for e in els] == ["bullet"] * 4
    assert [e.data["state"] for e in els] == ["open", "completed", "migrated", "question"]
    assert els[0].data["item"] == "open task"


def test_bullet_glyph_needs_a_space():
    els = parse_lines([_line("xylophone practice")])  # not a bullet
    assert els[0].kind == "text"


def test_tag_only_line_vs_text_with_tags():
    els = parse_lines([_line("#AI #[Data Science]"), _line("some prose #backend here")])
    assert els[0].kind == "tags"
    assert els[0].data["tags"] == ["AI", "Data Science"]
    assert els[1].kind == "text"
    assert els[1].data["tags"] == ["backend"]


def test_temporal_tags():
    els = parse_lines([
        _line("[DUE: 2026-09-14]"),
        _line("[EVENT: 2026-09-18 14:00]"),
        _line("[RANGE: 2026-09-12 -> 2026-09-19]"),
    ])
    kinds = [(e.data["type"], e.data.get("start"), e.data.get("end")) for e in els]
    assert kinds == [
        ("due", "2026-09-14", None),
        ("event", "2026-09-18 14:00", None),
        ("range", "2026-09-12", "2026-09-19"),
    ]


def test_reference_line():
    els = parse_lines([_line("-> [Research:P017:AUTH]")])
    assert els[0].kind == "reference"
    assert (els[0].data["document"], els[0].data["page"], els[0].data["anchor"]) == (
        "Research", "P017", "AUTH",
    )


def test_contact_block():
    els = parse_lines([
        _line("+ CONTACT"),
        _line("Jane Smith"),
        _line("jane@example.com  555-123-4567"),
        _line("Acme Corp"),
        _line("+---------------+"),
        _line("back to notes"),
    ])
    contact = next(e for e in els if e.kind == "contact")
    assert contact.data["name"] == "Jane Smith"
    assert contact.data["email"] == "jane@example.com"
    assert "555-123-4567" in contact.data["phone"]
    assert contact.data["organization"] == "Acme Corp"
    assert els[-1].kind == "text" and els[-1].text == "back to notes"


def test_all_bullet_glyphs_map_to_states():
    assert set(BULLET_STATES.values()) >= {
        "open", "completed", "migrated", "scheduled", "note", "event",
        "important", "question",
    }
