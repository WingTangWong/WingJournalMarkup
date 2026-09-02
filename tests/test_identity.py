from wingjournal.storage import Store, resolve_identity


def test_new_page_when_no_ids(tmp_path):
    with Store(tmp_path / "s") as store:
        r1 = resolve_identity(store)
        r2 = resolve_identity(store)
        assert r1.created and r2.created
        assert r1.source == "new"
        assert r1.page.uuid != r2.page.uuid  # no evidence -> distinct pages


def test_same_handwritten_id_resolves_to_one_page(tmp_path):
    with Store(tmp_path / "s") as store:
        r1 = resolve_identity(store, page_id_explicit="P017", topic_tags=["AI"])
        r2 = resolve_identity(store, page_id_explicit="P017", topic_tags=["ML"])
        assert r1.created and not r2.created
        assert r1.page.uuid == r2.page.uuid
        assert r2.source == "handwritten_id"
        assert set(store.get_page(r1.page.uuid).topic_tags) == {"AI", "ML"}


def test_machine_id_beats_handwritten(tmp_path):
    with Store(tmp_path / "s") as store:
        resolve_identity(store, page_id_machine="M-1", page_id_explicit="P017")
        # a later capture with only the machine id still finds it
        r = resolve_identity(store, page_id_machine="M-1")
        assert not r.created and r.source == "machine_id"


def test_conflicting_handwritten_id_is_surfaced(tmp_path):
    with Store(tmp_path / "s") as store:
        r1 = resolve_identity(store, page_id_machine="M-1", page_id_explicit="P017")
        r2 = resolve_identity(store, page_id_machine="M-1", page_id_explicit="P019")
        assert r2.page.uuid == r1.page.uuid
        assert any(c.kind == "page_id" for c in r2.conflicts)
        assert len(store.conflicts(r1.page.uuid)) == 1
        # the physical page content is not silently rewritten
        assert store.get_page(r1.page.uuid).page_id_explicit == "P017"
