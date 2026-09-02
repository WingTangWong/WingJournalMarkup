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
- Contributing + dev setup: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)
- Ready-to-print writing sheets: [`samples/`](samples/)

## Status

Early. The ingestion front-end runs end to end: acquire → preprocess →
ArUco + undecoded-square detection → ranked page-boundary hypotheses →
orientation resolution → perspective normalization + upright rotation →
`Capture` JSON. It is strong with 3–4 corner markers and degrades (for now, to a
whole-frame guess) with fewer. Everything past normalization is roadmap.

| Stage | State |
|---|---|
| Capture sources (`FileSource`, `DirectorySource`) | ✅ done |
| Preprocess (grayscale / CLAHE / threshold / edges / quads) | ✅ done |
| ArUco detection + generation | ✅ done |
| Undecoded square-contour → `FiducialCandidate` | ✅ done |
| Multi-evidence boundary hypotheses + weighted scorer | ✅ done |
| Orientation resolution (marker IDs; text-baseline fallback) | ✅ done |
| Partial-marker frame (3-corner completion, markers ⊕ squares) | ✅ done |
| Structure-envelope hypotheses + iterative refinement | ✅ done |
| Perspective normalization (homography + warp + rotate) | ✅ done |
| Evaluation harness (`wingjournal eval`) + `--debug` overlays | ✅ done |
| Printable writing sheet + legend (PDF / PNG) | ✅ done |
| Real phone-photo test corpus | ⬜ roadmap |
| Literal-box masking, OCR, WJM markup parser | ⬜ roadmap |
| Page / document / semantic graphs, capture reconciliation | ⬜ roadmap |

Synthetic detection numbers (`wingjournal eval --cases 40`): 4-marker boundary
IoU ≈ 0.998 / orientation 100%; 3-marker ≈ 0.95; 2-marker ≈ 0.75; 1-marker ≈ 0.69.

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`opencv-python-headless` is pulled in automatically. On a Raspberry Pi / ARM host
the wheel is available from PyPI; no system build needed.

## Usage

```bash
# Print a blank writing sheet (corner markers + empty metadata block) and
# the markup legend, then write on the sheet by hand. Ready-made sheets are
# in samples/; make-sheet writes .pdf or .png depending on the extension.
wingjournal make-sheet   --out writing-sheet.pdf --pages 5 --ruled
wingjournal make-sheet   --out writing-sheet.png
wingjournal make-legend  --out legend.pdf

# Run the ingestion pipeline: writes ./out/normalized/<name>.png
# and a JSON Capture sidecar to ./out/captures/<name>.json
wingjournal ingest photo.jpg --out out

# Add --debug for per-stage overlays in ./out/debug/
wingjournal ingest photo.jpg --out out --debug

# A whole directory of scans
wingjournal ingest ./scans/ --out out --recursive

# Score boundary + orientation detection on a synthetic corpus
wingjournal eval --cases 40 --verbose

# Generate a synthetic WJM page image to experiment with
wingjournal make-test-page --out page.png --warp --seed 1

# List ArUco dictionaries you can pass to --dict
wingjournal dictionaries
```

Hypothesis scoring weights can be overridden with `--weights weights.json`
(any subset of the `ScoringWeights` fields). `show-page` and `history` are
recognized but not implemented yet — they need the page/document graph (see the
roadmap).

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
