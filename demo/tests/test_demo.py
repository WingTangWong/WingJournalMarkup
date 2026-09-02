import io
import re


def _upload(client, data: bytes, name: str = "p.png"):
    return client.post(
        "/upload",
        data={"image": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def test_pages_render(client):
    assert client.get("/").status_code == 200
    assert client.get("/graph").status_code == 200
    g = client.get("/api/graph").get_json()
    assert g == {"nodes": [], "edges": [], "documents": {}, "layout": {}, "orphans": []}


def test_upload_ingests_and_shows_capture(client, page_png):
    r = _upload(client, page_png)
    assert r.status_code == 200
    m = re.search(rb"/capture/([0-9a-f-]{36})", r.data)
    assert m
    cid = m.group(1).decode()

    cap = client.get(f"/capture/{cid}")
    assert cap.status_code == 200
    body = cap.data.decode()
    for section in ("Normalized page", "Detection provenance", "Structured content"):
        assert section in body

    pdf = client.get(f"/capture/{cid}/pdf")
    assert pdf.status_code == 200
    assert pdf.data[:5] == b"%PDF-"


def test_scan_is_archived_with_uuid_and_timestamp(client, page_png, tmp_path):
    _upload(client, page_png, "myscan.png")
    scans = list((tmp_path / "data" / "scans").iterdir())
    assert len(scans) == 1
    assert re.match(r"[0-9a-f-]{36}_\d{8}T\d{6}Z\.png", scans[0].name)


def test_graph_grows_and_reports_orphans(client, page_png):
    _upload(client, page_png)
    _upload(client, page_png)
    g = client.get("/api/graph").get_json()
    assert len(g["nodes"]) == 2
    # no readable page ids on the synthetic corpus -> both orphaned
    assert len(g["orphans"]) == 2


def test_blob_route_serves_the_normalized_image(client, page_png):
    r = _upload(client, page_png)
    cid = re.search(rb"/capture/([0-9a-f-]{36})", r.data).group(1).decode()
    cap = client.get(f"/api/capture/{cid}").get_json()
    img = client.get(f"/blob/{cap['normalized_blob']}")
    assert img.status_code == 200
    assert img.data[:8] == b"\x89PNG\r\n\x1a\n"
