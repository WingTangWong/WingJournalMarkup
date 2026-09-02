from wingjournal.recognition.tags import (
    PageMetadata,
    parse_metadata_cells,
    parse_reference,
    parse_tag,
    parse_tags,
    split_qualified,
)


def test_parse_tag_single():
    assert parse_tag("#auth") == "auth"
    assert parse_tag("#[Data Science]") == "Data Science"
    assert parse_tag("  #P017 ") == "P017"
    assert parse_tag("plain") is None
    assert parse_tag("#") is None


def test_parse_tags_in_line():
    assert parse_tags("#AI #[Data Science] and #python") == ["AI", "Data Science", "python"]
    assert parse_tags("no tags here") == []
    assert parse_tags("#a #a") == ["a", "a"]  # duplicates kept


def test_parse_metadata_cells_example_from_spec():
    md = parse_metadata_cells(
        ["#Research", "#P017", "#AI #[Data Science]"],
        ["#P016", "", "#P027", "#P018"],
    )
    assert md == PageMetadata(
        document_id="Research",
        page_id="P017",
        topic_tags=["AI", "Data Science"],
        left="P016",
        above=None,
        below="P027",
        right="P018",
    )


def test_parse_metadata_cells_all_blank():
    md = parse_metadata_cells([], [])
    assert md == PageMetadata()


def test_split_qualified():
    assert split_qualified("Research:P017:AUTH") == ("Research", "P017", "AUTH")
    assert split_qualified("P017:AUTH") == (None, "P017", "AUTH")
    assert split_qualified("AUTH") == (None, None, "AUTH")
    assert split_qualified("Research : P017 : #[Auth Service]") == (
        "Research", "P017", "Auth Service",
    )


def test_parse_reference_forms():
    assert parse_reference("-> [#AUTH]").anchor == "AUTH"
    assert parse_reference("REF: #AUTH").anchor == "AUTH"
    r = parse_reference("see -> [Research:P017:AUTH] for details")
    assert (r.document, r.page, r.anchor) == ("Research", "P017", "AUTH")
    assert parse_reference("-> [#[Auth Service]]").anchor == "Auth Service"
    assert parse_reference("just prose") is None
