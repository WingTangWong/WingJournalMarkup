# WJM Roadmap

Milestones for the CLI edition. Each milestone is a shippable slice with tests.
GitHub milestones + issues mirror this file; this file is the source of truth for
intent, the issues track execution.

Legend: ✅ done · 🚧 in progress · ⬜ not started

The `M` track is the ingestion pipeline; the `T` track is authoring tools that
help a person produce scannable pages.

---

## T1 – Printable authoring tools ✅

**Goal:** get a usable WJM page onto paper without proprietary stationery.

- ✅ `wingjournal make-sheet` — blank writing sheet: four corner ArUco
      markers (`DICT_4X4_50`, IDs 0/1/2/3 = TL/TR/BR/BL) at a physically
      controlled size, their outer edge on the ¼-inch print margin so they
      hug the paper corners, plus an empty two-row metadata block with ¼-inch
      rows sitting in the span between the top two markers, vertically centred
      within the marker height (spec §11); `--paper letter|a4|legal`,
      `--pages`, `--marker-mm`, `--margin-mm`, `--ruled`; writes PDF or PNG by
      output extension. Ready-made Letter sheets in `samples/`
- ✅ `wingjournal make-legend` — one-page markup cheat-sheet PDF distilled from
      spec §10–22 (tags, bullet states, boxes, literal region, arrows, temporal,
      contacts, anchors/references, fiducials)
- ✅ Layout math in mm → px (`templates/geometry.py`); pages rendered as the
      exact raster the detector will see, so `make-sheet` output round-trips
      through `ingest`
- ⬜ Sticker-sheet / cut-guide PDF for adhesive corner markers
- ⬜ Overlay-template mode (fiducials only, handwritten page identity)

---

## D1 – Demo web app ✅

**Goal:** a browsable end-to-end demo of the pipeline.

- ✅ `demo/` — Flask app: upload photos → ingest → persist (SQLite + blob store +
      `data/scans/<uuid>_<timestamp>` archive)
- ✅ Colour-coded structured capture view (identity, provenance, literal regions,
      conflicts, parsed elements)
- ✅ Reassembled PDF download (overlaid scan + text sheet)
- ✅ Differential update on re-upload of a known page
- ✅ Graph explorer (cytoscape.js): spatial links, document grouping, orphans,
      pan/zoom, spatial-grid layout
- ⬜ Live re-render while a batch ingests; auth; deploy target (out of scope for
      a local demo)

---

## D2 – Mobile capture demo ✅

**Goal:** a phone-first capture front-end that auto-shoots when a sheet is framed.

- ✅ `MobileDeviceDemo/` — static web app, no build step; OpenCV.js (WASM) +
      ArUco (`DICT_4X4_50`) in a Web Worker so the 11 MB compile and the
      per-frame detection never stall the camera preview
- ✅ Live camera + guide overlay; auto-capture when marker ids 0/1/2/3 all sit
      inside the guide and hold ~450 ms (or a manual shutter)
- ✅ On capture: full-res photo **+** perspective-rectified page (mirrors
      `wingjournal.vision.{aruco,boundary,rectify}` — outer-corner page quad,
      1600 px long side, aspect clamp) **+** a JSON marker sidecar, all
      downloadable
- ✅ `vendor/opencv.js` committed; deployed on GitHub Pages
      (`wingtangwong.github.io/WingJournalMarkup/MobileDeviceDemo/`);
      `serve.py` (HTTP / HTTPS) for local / LAN
- ⬜ Feed a capture straight into `ingest` / the store; batch mode

---

## M0 – Project scaffold ✅

**Goal:** a runnable, tested Python package and a public repo.

- ✅ `src/` layout, `pyproject.toml`, `wingjournal` / `wjm` entry points
- ✅ `argparse` CLI with subcommands + stubs
- ✅ virtualenv + `opencv-python-headless` + `numpy`
- ✅ `pytest` suite, `ruff` lint, GitHub Actions CI
- ✅ Assessment + roadmap docs, README, contributing guide
- ✅ Public GitHub repository

---

## M1 – Acquire and rectify a page ✅

**Goal:** turn one photo into a perspective-normalized page image + a `Capture`
record.

- ✅ `CaptureSource` abstraction; `FileSource`, `DirectorySource`
- ✅ Preprocess: grayscale, CLAHE, adaptive threshold, Canny, quad extraction
- ✅ ArUco detection (OpenCV ≥ 4.7 OO API) + marker generation
- ✅ Marker-role assignment (TL/TR/BR/BL from the constellation)
- ✅ Page-boundary hypothesis: marker constellation → largest quad → full frame
- ✅ Perspective normalization: ordered corners → homography → `warpPerspective`
- ✅ `Capture` model + JSON sidecar output
- ✅ Synthetic page generator (`make-test-page`) for tests and demos
- ✅ `wingjournal ingest` command

**Exit criteria (met):** a warped synthetic page is detected (4/4 markers) and
rectified so the detected boundary maps to the normalized image corners within
1px; no-fiducial input falls back without crashing.

---

## M2 – Robust boundary & orientation ✅ (real-photo validation still open)

**Goal:** the spec's graceful-degradation ladder (sections 9, 25–33), measured.

- ✅ Undecoded square-contour catalog → `FiducialCandidate` with `inferred_role`
      and confidence (`vision/fiducial_candidates.py`, spec §25–26)
- ✅ Multi-evidence page-hypothesis model + weighted scorer (`vision/hypothesis.py`,
      spec §30–31): 5 signals, 2 penalties, weights in `ScoringWeights`
      (JSON-loadable via `--weights`)
- ✅ Orientation Tier A/B — decoded marker IDs in the constellation → exact
      0/90/180/270 (works with ≥2 markers when unambiguous)
- ✅ Orientation Tier F — dominant text-line direction (`flip_ambiguous` axis)
- ✅ Partial-marker geometry: 3-corner parallelogram completion + markers ⊕
      square candidates, greedily role-matched, for the missing corners (spec §32
      tier C)
- ✅ Structure-envelope hypotheses (`vision/envelope.py`, spec §27–28 / tier G):
      content bbox + margin, rotated-rect, and top-anchored page-aspect frames
- ✅ Iterative refinement loop (`select_boundary`, spec §34): re-detects corner
      squares near the provisional frame with a relaxed threshold, ≤3 passes,
      stops on score/convergence
- ✅ **Evaluation harness** (`wingjournal eval`): labelled synthetic corpus,
      boundary IoU + orientation accuracy bucketed by marker count; CI
      regression test on the 4- / 3- / 2-marker buckets
- ✅ `--debug` output: per-stage overlays
- ⬜ Orientation Tier E — metadata-block-as-TOP (needs the block detector, M3)
- ⬜ Real phone-photo corpus + documented numbers (#15; blocked on having scans)

**Current synthetic numbers** (`wingjournal eval --cases 40`): 4-marker IoU
≈ 0.998 / orientation 1.00; 3-marker ≈ 0.95 / 1.00; 2-marker ≈ 0.74–0.77;
1-marker ≈ 0.69. Orientation for 2/1-marker is ~0.6 (text-baseline can't
resolve the 180° flip — that's Tier E, M3).

**Exit criteria:** ≥ 0.95 boundary IoU and ≥ 98% orientation accuracy on the
4-marker set *(met, synthetic)*; graceful, **measured** degradation for 3/2/1/0
markers *(met — 3-marker ~0.95, 2/1-marker ~0.7–0.8, all well above the
full-frame ~0.5 floor)*; documented numbers for the real-photo subset
*(pending #15)*.

---

## M3 – Persistence & page identity 🚧

**Goal:** captures accumulate against persistent pages on disk.

- ✅ Normalized-coordinate spec (`docs/COORDINATES.md`): origin, axes, fixed
      target size / aspect clamp; `rectify` now produces a deterministic size so
      repeated captures line up (resolves ASSESSMENT §3.10)
- ✅ SQLite storage + content-addressed blob store (`wingjournal/storage/`):
      `Document` / `Page` / `Capture` / `PageRelationship` / `Conflict`;
      `wingjournal ingest --store DIR`
- ✅ Segmented metadata-block detection (`recognition/metadata_block.py`, spec
      §11): outer box, row divider, per-row cell grid (3 / 4) — geometry only;
      also drives orientation **Tier E** (2/1-marker orientation 0.6 → 0.8)
- ✅ Tag grammar parser (`recognition/tags.py`, spec §10, §19): `#term` /
      `#[term with spaces]`, metadata cells → `PageMetadata`, `document:page:anchor`
      references. Pure + fully tested; ready for OCR output in M4
- ✅ Page-identity resolution ladder + `Conflict` model (`storage/identity.py`,
      spec §39, §46): machine id > handwritten id > new; contradictions stored,
      never auto-resolved. (Ladder is tested with synthetic ids; the pipeline
      can't supply ids until OCR, so today every capture → a fresh page.)
- ✅ `wingjournal show-page <ref>` and `wingjournal history <ref>`
- ⬜ Metadata-block **cell text** → real `Page` identity/tags (needs OCR, M4)
- ⬜ Visual-match identity for markerless re-scans (spec §39 rung 4)
- ⬜ Spatial-graph document-membership propagation (moves toward M7)

**Exit criteria:** ingesting the same page twice updates one `Page` with two
`Capture`s *(done — with OCR reading the page id, `ingest --store` twice ->
one Page; tested in test_ocr_integration.py)*; a machine/handwritten ID mismatch
produces a stored conflict *(done, tested)*.

---

## M4 – Handwriting / text recognition 🚧

**Goal:** text regions become strings, with graceful absence.

- ✅ `TextRecognizer` interface (`recognition/text/`): region → text + per-word
      bbox + confidence
- ✅ Tesseract backend (`text/tesseract.py`) — local, offline; optional
      (`pip install .[ocr]` + system `tesseract-ocr`). Default recognizer is
      `auto` (use it if the binary is present, else fall back)
- ✅ `NullRecognizer` — always available, returns an "unrecognized region"
      placeholder; the pipeline always completes with recognition disabled
- ✅ Projection-profile line / word segmentation (`text/segment.py`)
- ✅ Wired end to end: metadata-block cells → `PageMetadata` → identity
      resolution; `ingest --recognizer auto|tesseract|none`
- ⬜ OCR of node bodies / bullet lines / freeform regions (feeds M5)
- ⬜ Optional hosted-HTR backend behind a flag (interface is ready; not built —
      local-only stays the default)
- ⬜ Real phone-photo accuracy numbers (shares #15's blocker)

**Exit criteria:** metadata block recognized end to end with the local backend
*(done, CI installs tesseract)*; pipeline still completes with recognition
disabled *(done, default on this dev box)*.

---

## M5 – WJM markup parser 🚧

**Goal:** recognized content → typed elements.

- ✅ Canonical reference enclosure: `-> [#ANCHOR]` / `REF: #ANCHOR`,
      `document:page:anchor` (`recognition/tags.py`, done in M3; resolves
      ASSESSMENT §3.2)
- ✅ Literal-asset detection (`vision/literal_box.py`, spec §16): four solid
      diagonal corner fills → `LiteralAsset`; interior masked **before** the
      recognition stages (§36) and stored as its own blob
- ⬜ Ordinary + segmented box detection → `DiagramNode` (title/body split, spec §14–15)
- ⬜ Tag scoping rules (tag inside a box belongs to that node, spec §14)
- ⬜ Bullet-glyph recognition + state vocabulary with per-glyph confidence (spec §18)
- ⬜ Anchors + references, local / cross-page / cross-document resolution (spec §19)
- ⬜ Temporal tags `[DUE:]` / `[EVENT:]` / `[RANGE:]` → `TemporalTag` (spec §21)
- ⬜ Contact blocks → `Contact` (spec §22)

**Exit criteria:** the canonical page example (spec §40) parses to the expected
element tree.

---

## M6 – Diagram graph extraction ⬜

**Goal:** hand-drawn boxes-and-arrows → nodes and edges.

- ⬜ Long-line detection; solid vs. dashed discrimination
- ⬜ Arrowhead detection + direction classification (`->`, `<-`, `<->`, `--`, `- -`)
- ⬜ Endpoint-to-node attachment by proximity to node boundary (spec §17)
- ⬜ Edge labels (text near the line midpoint)
- ⬜ `DiagramEdge` emission with provenance

**Exit criteria:** a 3-node / 2-edge hand-drawn diagram photo yields the correct
directed graph.

---

## M7 – Graphs: pages, documents, semantics ⬜

**Goal:** the three graph domains (spec §2), kept distinct.

- ⬜ Physical page graph: reciprocal relationship inference with provenance (spec §13)
- ⬜ Document membership propagation through connected components (spec §12);
      explicit vs. resolved kept separate; conflicting explicit IDs → conflict
- ⬜ Semantic content graph: nodes/edges/tags/anchors/references/contacts/temporal
- ⬜ Anchor + reference resolution across the corpus
- ⬜ Query commands: `wingjournal doc <id>`, `wingjournal tag <name>`,
      `wingjournal refs <ref>`

**Exit criteria:** a 3-page document linked only by spatial metadata resolves to
one document; a relationship contradiction is reported, not hidden.

---

## M8 – Capture reconciliation & versioning ⬜

**Goal:** a re-scan updates objects instead of duplicating them (spec §37–38).

- ⬜ Normalized-coordinate image diff (unchanged / added / removed regions)
- ⬜ Element matching: position + text similarity + neighbourhood + prior bbox
- ⬜ Semantic reconciliation: bullet state transitions, node body edits, edge
      add/remove, metadata changes — as updates to existing UUIDs
- ⬜ `CaptureDiff` model + `wingjournal diff <ref>` output
- ⬜ Reconciliation confidence + review list for low-confidence matches

**Exit criteria:** capturing a page, ticking one bullet, and re-capturing yields
`BulletItem <uuid>: open → completed` — not a second task.

---

## Cross-cutting (ongoing)

- Real-scan test corpus growth every milestone from M2
- `--debug` overlays for every vision stage
- Config file for tunable weights/thresholds
- Docs kept in step with behaviour
- Every automated commit pair-programmed and attributed (see CONTRIBUTING)
