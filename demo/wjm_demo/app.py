"""Flask app for the WJM demo. `python -m demo.run` (or `flask --app
demo.wjm_demo.app run`)."""

from __future__ import annotations

import io
import os
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from wjm_demo.graph import build_graph
from wjm_demo.highlight import element_view, summarize_elements
from wjm_demo.ingest import ingest_upload
from wjm_demo.reassemble import build_pdf
from wjm_demo.store import DemoStore

_DATA = Path(os.environ.get("WJM_DEMO_DATA", Path(__file__).resolve().parents[1] / "data"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "wjm-demo"
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    store = DemoStore(_DATA)
    app.config["STORE"] = store

    @app.get("/")
    def index():
        scans = store.scans()
        pages = store.pages()
        return render_template("index.html", scans=scans, pages=pages,
                               n_captures=len(store.all_captures()))

    @app.post("/upload")
    def upload():
        files = request.files.getlist("image")
        if not files or all(not f.filename for f in files):
            flash("choose an image first")
            return redirect(url_for("index"))
        last = None
        for f in files:
            if not f.filename:
                continue
            try:
                res = ingest_upload(store, f.filename, f.read())
                last = res["capture"].uuid
                msg = f"{f.filename}: page {res['page'].uuid[:8]}"
                if res["diff"]:
                    msg += " - differential update captured"
                flash(msg)
            except Exception as exc:  # demo: show, don't 500
                flash(f"{f.filename}: {exc}")
        return redirect(url_for("capture", capture_uuid=last) if last else url_for("index"))

    @app.get("/capture/<capture_uuid>")
    def capture(capture_uuid: str):
        cap = store.wjm.get_capture(capture_uuid)
        if cap is None:
            abort(404)
        page = store.wjm.get_page(cap["page_uuid"]) if cap.get("page_uuid") else None
        rels = store.wjm.relationships_for_page(page.uuid) if page else []
        conflicts = store.wjm.conflicts(page.uuid) if page else []
        elements = [element_view(e) for e in cap.get("detected_elements", [])]
        return render_template(
            "capture.html", cap=cap, page=page, elements=elements,
            summary=summarize_elements(cap.get("detected_elements", [])),
            rels=rels, conflicts=conflicts,
            diff=store.get_diff(capture_uuid),
            scan=store.scan_for_capture(capture_uuid),
            captures_for_page=(store.wjm.captures_for_page(page.uuid) if page else []),
        )

    @app.get("/capture/<capture_uuid>/pdf")
    def capture_pdf(capture_uuid: str):
        try:
            data = build_pdf(store, capture_uuid)
        except KeyError:
            abort(404)
        return send_file(io.BytesIO(data), mimetype="application/pdf",
                         as_attachment=True, download_name=f"wjm-{capture_uuid[:8]}.pdf")

    @app.get("/blob/<digest>")
    def blob(digest: str):
        try:
            data = store.wjm.get_blob(digest)
        except FileNotFoundError:
            abort(404)
        return send_file(io.BytesIO(data), mimetype="image/png")

    @app.get("/graph")
    def graph():
        return render_template("graph.html")

    @app.get("/api/graph")
    def api_graph():
        return jsonify(build_graph(store))

    @app.get("/api/capture/<capture_uuid>")
    def api_capture(capture_uuid: str):
        cap = store.wjm.get_capture(capture_uuid)
        if cap is None:
            abort(404)
        return jsonify(cap)

    @app.template_filter("shortuuid")
    def _short(u):  # noqa: ANN001
        return (u or "")[:8]

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
