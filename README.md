# Wing Journal Markup (WJM)

A paper-native structured information system. Write naturally on ordinary paper;
WJM scans the page, identifies it, normalizes perspective, parses the WJM markup,
and represents it digitally as a persistent structured object.

> **The page is a persistent structured object. A photograph or scan is only an
> observation of that object.**

This repository is the **CLI edition**: Python + OpenCV, command-line only for now.

- Full design: [`docs/SPEC-v0-draft.md`](docs/SPEC-v0-draft.md)
- Independent review of that design: [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md)
- Milestones and tasks: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Normalized page coordinates: [`docs/COORDINATES.md`](docs/COORDINATES.md)
- Contributing + dev setup: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)
- Ready-to-print writing sheets: [`samples/`](samples/)
- Demo web app (upload → structured view → PDF → graph explorer): [`demo/`](demo/)

## Status

Early. The ingestion front-end runs end to end: acquire → preprocess →
ArUco + undecoded-square detection → ranked page-boundary hypotheses →
orientation resolution → perspective normalization → metadata-block detection →
persist to a store (SQLite + content-addressed blobs). It is strong with 3–4
corner markers and degrades gracefully with fewer. OCR and the markup parser are
the next milestone.

| Stage | State |
|---|---|
| Capture sources, preprocess, ArUco detect/generate | ✅ done |
| Boundary: undecoded squares, weighted hypothesis scorer, iterative refinement | ✅ done |
| Partial-marker frame + content-envelope fallback | ✅ done |
| Orientation: marker IDs → metadata-block → text baseline | ✅ done |
| Perspective normalization → fixed normalized coordinates ([`docs/COORDINATES.md`](docs/COORDINATES.md)) | ✅ done |
| Segmented metadata-block detection (geometry) | ✅ done |
| Tag grammar (`#term` / `#[term with spaces]` / references) | ✅ done |
| SQLite + blob store; `ingest --store`, `show-page`, `history` | ✅ done |
| Text recognition: `TextRecognizer` + Tesseract backend + null fallback | ✅ done |
| Metadata cells → `PageMetadata` → page-identity ladder + conflicts | ✅ done |
| Literal image regions (four-corner mounts): detect, mask, store (spec §16) | ✅ done |
| Evaluation harness + `--debug` overlays; writing-sheet / legend PDF·PNG | ✅ done |
| Boxes → nodes, bullets, temporal / contact / anchor parsing | ⬜ next (M5) |
| Diagram graph, semantic graphs, capture reconciliation | ⬜ roadmap |

Synthetic detection numbers (`wingjournal eval --cases 40`): 4-marker boundary
IoU ≈ 0.998 / orientation 100%; 3-marker ≈ 0.95 / 100%; 2-marker ≈ 0.75 / 80%;
1-marker ≈ 0.69 / 80%.

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`opencv-python-headless` is pulled in automatically. On a Raspberry Pi / ARM host
the wheel is available from PyPI; no system build needed.

Text recognition is optional. For it, install the extra **and** the system
Tesseract binary:

```bash
pip install -e ".[ocr]"
sudo apt install tesseract-ocr        # or: brew install tesseract
```

Without it, `ingest` still runs — metadata cells just come back unread.

## Usage

```bash
# Print a blank writing sheet (corner markers + empty metadata block) and
# the markup legend, then write on the sheet by hand. Ready-made sheets are
# in samples/; make-sheet writes .pdf or .png depending on the extension.
wingjournal make-sheet   --out writing-sheet.pdf --pages 5 --ruled
wingjournal make-sheet   --out writing-sheet.png
wingjournal make-legend  --out legend.pdf

# Run the ingestion pipeline: writes ./out/normalized/<name>.png + a JSON
# Capture sidecar, and (with --store) persists to a SQLite + blob store
wingjournal ingest photo.jpg --out out --store wjm-store
wingjournal ingest ./scans/  --out out --store wjm-store --recursive
wingjournal ingest photo.jpg --out out --debug        # per-stage overlays

# Inspect the store
wingjournal show-page <page-uuid-or-prefix> --store wjm-store
wingjournal history  <page-uuid-or-prefix> --store wjm-store

# Score boundary + orientation detection on a synthetic corpus
wingjournal eval --cases 40 --verbose

wingjournal make-test-page --out page.png --warp --seed 1
wingjournal dictionaries
```

Hypothesis scoring weights can be overridden with `--weights weights.json` (any
subset of the `ScoringWeights` fields). Until OCR lands (M4) the pipeline can't
read handwritten page ids, so every capture resolves to a fresh page.

## Fiducials

Print four ArUco markers (default dictionary `DICT_4X4_50`, IDs 0/1/2/3 for
TL/TR/BR/BL) and place them near the page corners. Alignment does not need to be
precise — the marker *constellation* matters more than any single sticker's
rotation. Pages remain ingestible with fewer markers or none, at lower
confidence.

## Tests

```bash
pytest
ruff check src tests
```

## License

Not licensed yet — all rights reserved for now. A license will be added before
any external contributions are accepted.
