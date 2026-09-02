# WJM demo web app

A Flask front-end over the `wingjournal` pipeline: upload page photos, see the
structured extraction (colour-coded), download a reassembled PDF, and explore the
page graph.

## Run

```bash
# from the repo root, in the project venv
pip install -e ".[ocr]"                 # the wingjournal package + tesseract extra
pip install -r demo/requirements.txt    # flask
sudo apt install tesseract-ocr          # optional but needed for text extraction

python demo/run.py                      # -> http://127.0.0.1:5000
```

Set `WJM_DEMO_DATA=/some/dir` to change where state lives (default `demo/data/`,
git-ignored).

## What it does

- **Upload** one or more images. Each is archived to
  `data/scans/<uuid>_<UTC timestamp>.<ext>`, ingested through the full pipeline
  (boundary → orientation → normalize → literal-region mask → metadata block →
  OCR → element parse), and persisted to `data/store/` (SQLite + content-addressed
  blob tree).
- **Capture view** (`/capture/<uuid>`): the normalized page, page metadata and
  identity, detection provenance, literal image regions, conflicts, and the
  parsed body as a colour-coded list — headings, notes, bullets (with journal
  state), tag lines, temporal markers, references, contact blocks.
- **Reassembled PDF** (`/capture/<uuid>/pdf`): page 1 is the normalized scan with
  the detected structure drawn on it; page 2 is a text sheet of the extraction.
- **Differential updates**: re-uploading a page whose `#PAGE` id is readable
  resolves to the same `Page`; the change from the previous capture (added /
  removed / re-stated elements, metadata moves) is stored and shown.
- **Graph explorer** (`/graph`): pages as nodes, `LEFT/ABOVE/BELOW/RIGHT` links
  as edges (explicit + inferred reciprocals), same-document grouping, orphans
  highlighted. Pan / zoom; toggle a spatial grid layout derived from the L/A/B/R
  relationships.

Without `tesseract-ocr` installed everything still runs — you just get the
geometry, provenance and images, with no body text or identity.

## Layout

```
demo/
  run.py                entry point
  wjm_demo/
    app.py              Flask routes
    store.py            DemoStore = wingjournal Store + demo.sqlite + scans/
    ingest.py           archive -> ingest -> persist -> relationship inference -> diff
    diff.py             capture-to-capture element / metadata diff
    graph.py            nodes / edges / orphans / grid layout
    reassemble.py       the downloadable PDF
    highlight.py        element -> CSS class / label
    templates/  static/
  tests/                pytest demo/tests   (needs flask)
```
