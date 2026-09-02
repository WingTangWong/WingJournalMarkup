from wingjournal.models import Capture, Conflict, Page, PageRelationship
from wingjournal.storage import Store


def test_blob_store_is_content_addressed(tmp_path):
    with Store(tmp_path / "s") as store:
        a = store.put_blob(b"hello wjm")
        b = store.put_blob(b"hello wjm")
        c = store.put_blob(b"different")
        assert a == b != c
        assert len(a) == 64
        assert store.get_blob(a) == b"hello wjm"
        assert store.blob_path(a).exists()


def test_page_and_capture_roundtrip(tmp_path):
    with Store(tmp_path / "s") as store:
        page = Page(page_id_explicit="P017", topic_tags=["AI"])
        store.upsert_page(page)

        cap = Capture(page_uuid=page.uuid, page_boundary_method="aruco_constellation",
                      page_boundary_confidence=0.95, orientation_degrees=0,
                      detected_fiducials=[])
        store.add_capture(cap)

        got = store.get_page(page.uuid)
        assert got.page_id_explicit == "P017"
        assert got.topic_tags == ["AI"]
        assert got.capture_uuids == [cap.uuid]

        caps = store.captures_for_page(page.uuid)
        assert len(caps) == 1
        assert caps[0]["page_boundary_method"] == "aruco_constellation"
        assert caps[0]["uuid"] == cap.uuid


def test_find_page_prefers_machine_id(tmp_path):
    with Store(tmp_path / "s") as store:
        store.upsert_page(Page(page_id_explicit="P017", page_id_machine="M-1"))
        assert store.find_page(page_id_machine="M-1").page_id_explicit == "P017"
        assert store.find_page(page_id_explicit="P017").page_id_machine == "M-1"
        assert store.find_page(page_id_explicit="nope") is None


def test_relationships_and_conflicts(tmp_path):
    with Store(tmp_path / "s") as store:
        p1, p2 = Page(), Page()
        store.upsert_page(p1)
        store.upsert_page(p2)
        store.add_relationship(PageRelationship(p1.uuid, p2.uuid, "RIGHT"))
        store.add_relationship(PageRelationship(p1.uuid, p2.uuid, "RIGHT"))  # dedup
        rels = store.relationships_for_page(p1.uuid)
        assert len(rels) == 1 and rels[0].relation == "RIGHT"

        store.add_conflict(Conflict(kind="page_id", detail="x vs y", page_uuid=p1.uuid))
        assert len(store.conflicts(p1.uuid)) == 1
        assert len(store.conflicts()) == 1


def test_store_reopens(tmp_path):
    path = tmp_path / "s"
    with Store(path) as store:
        page = Page(page_id_explicit="P1")
        store.upsert_page(page)
        uuid = page.uuid
    with Store(path) as store:
        assert store.get_page(uuid).page_id_explicit == "P1"
