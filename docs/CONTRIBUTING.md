# Contributing

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest
ruff check src tests
```

Python 3.10+. The vision stack is `opencv-python-headless` + `numpy`, both pulled
in by `pip install -e .`.

## Layout

```
src/wingjournal/
  capture/      capture sources (FileSource, DirectorySource, ...)
  vision/       OpenCV front-end: preprocess, aruco, boundary, fiducial_candidates,
                hypothesis (scorer + iterative select_boundary), envelope,
                orientation, rectify, debug, synthetic
  recognition/  tags (grammar), metadata_block (geometry); OCR-fed parsing lands M4
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
