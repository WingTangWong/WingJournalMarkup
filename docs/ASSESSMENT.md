# WJM Design Assessment

An independent review of [`SPEC-v0-draft.md`](SPEC-v0-draft.md), written to seed
the roadmap. It records what is strong, what is under-specified, what is risky,
and the scope decisions taken for the CLI edition.

## 1. Summary

The spec describes an ambitious, coherent system: paper as the primary UI, with a
vision pipeline that reconstructs a persistent page object from imperfect
photographs and reconciles repeated captures over time. The core principles
(section 50) are sound and mostly independent of each other, which makes the
project tractable to build in slices.

The main risk is **scope**. The spec spans document scanning, fiducial geometry,
handwriting recognition, a hand-drawn diagram language, a bullet-journal state
machine, three graph domains, and capture-diff versioning. Several of these are
research-grade problems on their own. The design is right to treat them as
layers; the roadmap must be disciplined about shipping the layers one at a time
with real tests.

## 2. Strengths

- **Observation vs. object.** Separating `Page` (persistent) from `Capture` (one
  observation) is the right backbone. It makes versioning, provenance, and
  conflict handling natural rather than bolted on.
- **Graceful degradation.** The confidence tiers (section 9) and orientation
  fallthrough (section 32) give a clear, testable ladder from "4 good markers" to
  "best-effort content envelope". This is a good fit for OpenCV primitives.
- **Provenance and conflict as first-class data** (sections 45–46). Storing
  explicit vs. resolved values separately, and surfacing conflicts instead of
  silently choosing, is the correct default for a system that ingests the same
  page many times.
- **Fiducials as evidence, not gospel** (sections 5, 7, 25–26). "The marker
  constellation matters more than sticker alignment" is a strong, implementable
  idea and avoids proprietary paper.
- **Literal regions** (section 16). An explicit escape hatch that is masked
  *before* semantic parsing is a clean way to stop drawings from becoming false
  graph nodes.
- **The canonical fallthrough** (sections 33–34) is already close to an
  implementable state machine.

## 3. Gaps and under-specified areas

Ordered roughly by how soon they block progress.

1. **Storage layer is unspecified.** Section 41–44 give models but no
   persistence. Needs a decision: SQLite + a content-addressed image/asset store
   is the obvious low-friction choice for a CLI. Graph queries are shallow enough
   that a full graph DB is not warranted yet.
2. **Page-ID namespace and reference grammar.** `document:page:anchor` is
   sketched (section 19) but the reference *enclosure* is explicitly deferred
   ("`-> [#AUTH]`" vs "`REF: #AUTH`"). The markup parser cannot be finished until
   this is pinned. Recommend fixing one canonical form early even if provisional.
3. **Marker → page-identity encoding.** Section 8 leaves the bit/ID encoding as
   "an implementation decision". Fine to defer, but the `Page.page_id_machine`
   path stays a stub until it is chosen.
4. **Handwriting recognition is a black box.** The spec assumes OCR/HTR exists
   (sections 6-in-48, 42) without naming an approach. Realistically this is an
   external dependency (a hosted HTR API, or Tesseract for the printed-ish
   subset, or a local model). The pipeline should isolate it behind an interface
   and degrade to "unrecognized text region" when absent.
5. **Metadata-block detection.** It is *defined* as a segmented box at the top of
   the page and used as strong orientation evidence (Tier E), but there is no
   detection algorithm. Detecting an N-cell segmented rectangle robustly is
   non-trivial and needs its own milestone.
6. **Diagram edge extraction.** Section 17's visual vocabulary (`--->`, `<-->`,
   `- - -`) requires line detection, arrowhead classification, dashed-vs-solid
   discrimination, and endpoint-to-node attachment. This is a milestone, not a
   task.
7. **Capture matching / semantic reconciliation** (sections 37–38) depends on
   normalized-coordinate stability, text similarity, and possibly handwriting
   similarity. Needs a concrete matching algorithm and a labelled test set.
8. **Bullet glyph recognition.** The state vocabulary (section 18) mixes glyphs
   that are hard to disambiguate in handwriting (`x` vs `*`, `>` vs `<`, `-` vs
   `–`). Needs a tolerance model and per-glyph confidence.
9. **No evaluation strategy.** There is no mention of ground-truth datasets,
   metrics (boundary IoU, orientation accuracy, parse F1), or regression corpora.
   This should exist from Milestone 2 onward.
10. **Coordinate-system definition.** "Normalized page coordinates" is used
    throughout but never defined (origin, units, aspect handling, DPI). Pin this
    before the parser stores any bbox.

## 4. Risks

- **Combinatorial hypothesis scoring** (section 31). The weighted scorer with
  "configurable weights" can become an untunable pile of magic numbers. Mitigate
  with a labelled boundary dataset and a small optimizer, and keep the number of
  signals small until they earn their place.
- **Iterative boundary refinement** (section 34) can oscillate or loop. Needs a
  fixed max pass count and a convergence check.
- **HTR cost and privacy.** If handwriting recognition is a hosted API, per-page
  cost and sending journal contents off-device both matter. Offer a local-only
  mode.
- **Over-fitting to synthetic pages.** The current tests use generated pages with
  crisp printed markers. Real phone photos add motion blur, page curl, shadows,
  lined/grid paper, and ballpoint ink. Real scans must enter the test corpus
  early.
- **Scope creep into a GUI.** The spec keeps saying "paper is the UI", but
  reviewing captures, conflicts, and the graph will create pressure for a viewer.
  Keep that out of this repo; emit JSON others can render.

## 5. Scope decisions for the CLI edition

- **Language / stack:** Python 3.10+, OpenCV (`opencv-python-headless`), NumPy.
  No GUI. `argparse` CLI, no heavy framework.
- **Package name:** `wingjournal`; commands `wingjournal` and `wjm`.
- **`src/` layout**, one package, submodules mirror the spec's section 47 layout
  but only the modules a milestone needs are created.
- **Storage (planned):** SQLite for models + a content-addressed store for raw /
  normalized images and literal assets.
- **HTR (planned):** pluggable interface, local option first, "unrecognized
  region" fallback always available.
- **Testing:** synthetic-page generator now; real-scan corpus + metrics from
  Milestone 2. `pytest` + `ruff` in CI.
- **Non-goals for now:** camera/scanner sources beyond `FileSource` /
  `DirectorySource`, the marker-identity bit encoding, any networked source, any
  viewer/GUI.

## 6. Recommended first three milestones

See [`ROADMAP.md`](ROADMAP.md) for the full plan. In short:

1. **M1 – Rectify a page** *(done)*: acquire → preprocess → ArUco → boundary →
   perspective normalization → `Capture` JSON.
2. **M2 – Robust boundary & orientation** *(mostly done)*: undecoded-square
   fiducial candidates, multi-evidence hypothesis scoring, marker-ID + text
   orientation, 3-corner completion, and a labelled corpus with IoU/accuracy
   metrics all landed. Still open: the structure-envelope boundary for the
   0–2-marker case, iterative refinement, metadata-block orientation (blocked on
   M3), and a real-photo corpus.
3. **M3 – Persistence & page identity**: SQLite storage, content-addressed image
   store, `Page`/`Capture` records, handwritten metadata-block detection and
   parsing, `show-page` / `history` commands.

Also shipped alongside M2: **T1 – printable authoring tools** (`make-sheet`,
`make-legend`), so a person can produce and hand-fill a scannable page.
