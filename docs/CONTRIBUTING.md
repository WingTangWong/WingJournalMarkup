# Contributing

## Dev setup

```bash
./setup.sh          # venv + system Tesseract + all deps + verify
./setup.sh --check  # ... and run the test suite
source .venv/bin/activate
```

`setup.sh` is idempotent: on an existing checkout it reuses the venv,
fast-forwards the repo (skipped if the tree is dirty), and re-syncs deps.
`./setup.sh --help` lists the flags (`--runtime`, `--no-ocr`, `--no-pull`,
`--python`).

Manual equivalent:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + OCR + pytest/ruff + flask
pip install -e .

pytest
ruff check src tests
```

Python 3.10+ (3.11/3.12 recommended for wheel coverage). The vision stack is
`opencv-python-headless` + `numpy` + `pillow` (`requirements.txt`); OCR adds
`pytesseract` plus the system `tesseract` binary (`requirements-ocr.txt`).

## Layout

```
src/wingjournal/
  capture/      capture sources (FileSource, DirectorySource, ...)
  vision/       OpenCV front-end: preprocess, aruco, boundary, fiducial_candidates,
                hypothesis (scorer + iterative select_boundary), envelope,
                orientation, rectify, literal_box, debug, synthetic
  recognition/  tags (grammar), metadata_block (geometry), metadata (cells->PageMetadata),
                text/ (TextRecognizer interface, Tesseract backend, null fallback, segment)
  storage/      SQLite + content-addressed blob store; identity resolution
  templates/    printable PDF / PNG: geometry, writing_sheet, legend
  eval/         evaluation harness: corpus, metrics, harness
  models/       dataclasses (Capture, Page, Document, PageRelationship, Conflict,
                DetectedMarker, FiducialCandidate, PageHypothesis, Orientation, ...)
  pipeline.py   the ingestion pipeline
  cli.py        argparse entry point (wingjournal / wjm)
tests/          pytest; conftest.py builds synthetic pages
docs/           SPEC-v0-draft, ASSESSMENT, ROADMAP, COORDINATES
```

New pipeline stages land as their own `vision/` or `recognition/` module with
tests, wired into `pipeline.py`. Keep stages independent and each one testable in
isolation. See `docs/ROADMAP.md` for what belongs where.

## Commits

- Small, focused commits; run `pytest` and `ruff` before committing.
- Conventional-ish subject lines (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

### Attribution

This project is developed as a pair. **Every commit and pull request** is
co-authored by Evangeline Speite. Commit messages end with:

```
Co-Authored-By: Evangeline Speite <contact@evangelinespeite.com>
```

That is the only trailer we add — no tooling or session metadata. A commit
message template is checked in; enable it locally with:

```bash
git config commit.template .gitmessage
```

## Tests

- Every vision stage needs a test against a synthetic page at minimum.
- `wingjournal eval` scores boundary IoU + orientation accuracy on a labelled
  synthetic corpus; `tests/test_eval.py` pins the 4-marker bucket so a
  regression fails CI. Extend the corpus (`eval/corpus.py`) when you touch
  detection, and add real-scan fixtures as they become available.
- `ruff check src tests` must be clean.
