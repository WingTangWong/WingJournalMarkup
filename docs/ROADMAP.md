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
      controlled size, plus an empty two-row metadata block (spec §11);
      `--paper letter|a4|legal`, `--pages`, `--marker-mm`, `--ruled`; writes
      PDF or PNG by output extension. Ready-made Letter sheets in `samples/`
- ✅ `wingjournal make-legend` — one-page markup cheat-sheet PDF distilled from
      spec §10–22 (tags, bullet states, boxes, literal region, arrows, temporal,
      contacts, anchors/references, fiducials)
- ✅ Layout math in mm → px (`templates/geometry.py`); pages rendered as the
      exact raster the detector will see, so `make-sheet` output round-trips
      through `ingest`
- ⬜ Sticker-sheet / cut-guide PDF for adhesive corner markers
- ⬜ Overlay-template mode (fiducials only, handwritten page identity)

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

## M3 – Persistence & page identity ⬜

**Goal:** captures accumulate against persistent pages on disk.

- ⬜ Coordinate-system spec: origin, units, aspect, DPI (resolves ASSESSMENT §3.10)
- ⬜ SQLite storage layer for `Document` / `Page` / `Capture` / `PageRelationship`
- ⬜ Content-addressed store for raw + normalized images and assets
- ⬜ Segmented metadata-block detection (top-of-page N-cell rectangle, spec §11)
- ⬜ Metadata parsing: `#term` / `#[term with spaces]` grammar (spec §10),
      document/page IDs, topic tags, L/A/B/R relationships
- ⬜ Page-identity resolution ladder (spec §39): machine ID > handwritten ID >
      spatial > visual match > new; **surface conflicts, never auto-resolve**
- ⬜ `wingjournal show-page <ref>` and `wingjournal history <ref>`

**Exit criteria:** ingesting the same page twice updates one `Page` with two
`Capture`s; a machine/handwritten ID mismatch produces a stored conflict.

---

## M4 – Handwriting / text recognition ⬜

**Goal:** text regions become strings, with graceful absence.

- ⬜ `TextRecognizer` interface; region → text + per-token confidence + bbox
- ⬜ Local backend (Tesseract for the printed-ish subset) as default
- ⬜ Optional hosted-HTR backend behind a flag; local-only mode is always valid
- ⬜ "Unrecognized text region" fallback element when no backend is available
- ⬜ Line/word segmentation on the normalized page

**Exit criteria:** metadata block + bullet lines from a real photo recognized
end-to-end with the local backend; pipeline still completes with recognition
disabled.

---

## M5 – WJM markup parser ⬜

**Goal:** recognized content → typed elements.

- ⬜ Canonical reference enclosure decided and documented (resolves ASSESSMENT §3.2)
- ⬜ Ordinary + segmented box detection → `DiagramNode` (title/body split, spec §14–15)
- ⬜ Tag scoping rules (tag inside a box belongs to that node, spec §14)
- ⬜ Bullet-glyph recognition + state vocabulary with per-glyph confidence (spec §18)
- ⬜ Anchors + references, local / cross-page / cross-document resolution (spec §19)
- ⬜ Temporal tags `[DUE:]` / `[EVENT:]` / `[RANGE:]` → `TemporalTag` (spec §21)
- ⬜ Contact blocks → `Contact` (spec §22)
- ⬜ Literal-asset detection (four diagonal black corners) + interior masking
      **before** parsing (spec §16, §36)

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
