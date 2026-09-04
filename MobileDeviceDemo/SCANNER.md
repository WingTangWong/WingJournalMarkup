# Web-demo document scanner — design note

Status: design, not built. Web demo only. The native app (TODO) is the
real-quality single-capture path; this note is about squeezing a usable
300 DPI page out of a low-resolution `getUserMedia` video stream.

## Why multi-shot

A `getUserMedia` frame is ~1–2 MP with no autofocus control and, on iOS, no
still-capture API. One frame of a whole Letter page ≈ 130 DPI and soft. So:
one wide **base** shot fixes geometry and the overall page, then up to five
**close-up** shots re-photograph the areas that carry detail, and we composite
them into a fixed 2550×3300 (Letter @ 300 DPI) canvas. Where a close-up landed,
the canvas has real resolution; elsewhere it stays soft — and the review screen
says which is which.

This replaces the current "lock → grab one video frame → hard sharpness gate"
flow, whose gate rejects real captures in dim light and still passes blurry ones.

## State machine

Data carried forward: `canvas` (2550×3300 working image), `pageHomography`
(base image px → canvas px), `targets` (list of canvas-space rects still needing
detail), `pass` (re-assess counter).

| State | Does | Transitions |
|---|---|---|
| **BOOT** | load OpenCV + OCR, open camera at max requested resolution | → SEEK_LOCK |
| **SEEK_LOCK** | per-frame anchor detection, guide overlay; need all four corner anchors in the guide and stable | 4-lock held ≥ 400 ms → BASE_BURST; otherwise loop |
| **BASE_BURST** | sample ~8 frames over ~600 ms, **no gating** | → BASE_SELECT |
| **BASE_SELECT** | Tenengrad focus score per frame over the page region; keep the best; rectify it into `canvas`, record `pageHomography` | → PLAN_TARGETS |
| **PLAN_TARGETS** | build the detail map on `canvas` (below); cluster into ≤ 5 rects sized to fill ~⅓–½ of a video frame at closer range | targets → TARGET_GUIDE(0); none → REVIEW |
| **TARGET_GUIDE(i)** | draw target *i* on the live video, positioned from the currently visible anchors; coach "move closer, keep **two** markers in view". Ready when the target roughly fills the frame, ≥ 1 anchor is visible, and local focus is trending up | ready → TARGET_BURST(i); anchor lost → coach |
| **TARGET_BURST(i)** | burst as BASE_BURST | → TARGET_SELECT(i) |
| **TARGET_SELECT(i)** | pick sharpest; register it to `canvas` (below); feathered-paste the target rect | i < last → TARGET_GUIDE(i+1); else → REASSESS |
| **REASSESS** | rebuild the detail map on the composite | soft targets remain **and** `pass < 2` → `pass++`, TARGET_GUIDE(0) with the new list; else → REVIEW |
| **REVIEW** | show the composite; list regions still soft; **Accept** / **Redo region** / **Restart** | Accept → EXTRACT; Redo → PLAN_TARGETS; Restart → SEEK_LOCK |
| **EXTRACT** | OCR + parse on the final composite (today's recognition tail) | done |

Global transitions: lost lock in any GUIDE state → coach, stay; user cancel →
SEEK_LOCK; hard error → fatal.

## Close-up registration

Goal: a homography mapping the close-up frame into `canvas` space.

1. **Two or more anchors with known page coordinates visible** → solve the
   homography directly from their corners (each ArUco gives 4 correspondences;
   the printed template's corner ids 0–3 and field ids 20–26 all have known
   canvas positions). This is the good path — a header close-up sees 2–4 field
   anchors easily.
2. **Fewer than two known anchors** (e.g. a mid-body close-up over plain ruled
   lines) → **ORB** feature match of the close-up against the corresponding
   crop of `canvas`; RANSAC homography from the matches. Any single visible
   anchor seeds the search region and sanity-checks scale.
3. **Box-corner clues** — the drawn field boxes and any hand-drawn boxes are
   strong rectangular features; their corners feed both the ORB match and a
   direct 4-point solve when a whole box is in view.

One anchor alone is **not** accepted for registration — hence the "keep two
markers in view" coaching.

## Detail map (PLAN_TARGETS / REASSESS)

Grid the canvas (~24×32 cells). A cell is a **target candidate** when it has
detail that a closer shot would improve. Signals, combined:

- **edge density** above a floor → there is ink/structure here (not blank paper);
- **resolved?** local high-frequency energy (Tenengrad or DoG response) below
  what a sharp capture at the current ground-sampling-distance would produce →
  the detail is under-resolved;
- **ArUco boxes** — a field-anchor box whose interior scored soft is always a
  candidate (that's the metadata we most want crisp);
- **ORB keypoint clusters** — dense keypoints that are low-contrast/blurred mark
  content worth a closer look;
- **box clues** — a detected rectangle (printed or hand-drawn) whose contents
  scored soft.

Merge adjacent candidate cells into axis-aligned rects; drop rects smaller than
~one grid cell; keep the five largest by (area × softness). Pad each rect so a
closer shot has margin, and clamp so it fits a video frame at ~2× the base
distance.

## Burst + selection

A "burst" is ~8 consecutive `requestAnimationFrame` grabs over ~600 ms. Score
each frame with **Tenengrad** (sum of Sobel gradient magnitude) over the region
of interest — whole page for the base, the target rect for a close-up. Keep the
single highest. **No absolute threshold at selection time** — we always keep the
best we got. Absolute sharpness only appears at REVIEW, to label regions and to
decide whether REASSESS bothers with another pass.

(Later, optional: align + average the top 2–3 frames to denoise. Not v1.)

## Compositing

`canvas` starts as the warped base (INTER_AREA when shrinking, INTER_LINEAR when
near 1:1 — never INTER_CUBIC, which overshoots hard edges and is what made the
current output "chunky"). Each close-up is warped into canvas space and pasted
over its target rect with a short (~12 px) alpha feather. No exposure matching in
v1; note it if seams are visible.

Output is 300 DPI *dimensionally*. It is honest about detail: the review screen
shades regions that never received a close-up.

## Build phases

- **A — base only.** ✅ BOOT → SEEK_LOCK → BASE_BURST → BASE_SELECT → REVIEW →
  EXTRACT. Best-of-burst single shot, INTER_AREA/LINEAR warp, no gate.
- **B — close-ups.** ✅ PLAN_TARGETS (metadata block + literals + paragraph-ish
  line clusters, ≤5) + the TARGET_* loop (live similarity-mapped target box,
  "keep two markers in view") + registration (`anchorHomography` from ≥2 shared
  ArUco ids, `orbHomography` fallback, both sanity-checked) + feathered
  `compositeInto` + a `finish` pass that recognises the composited canvas.
  Worker holds the canvas in a session across `analyze` → `composite`× →
  `finish`.
- **C — convergence.** ⬜ REASSESS with `pass < 2` cap and "still soft" review
  shading.

### Phase B — still to tune

- Target planning is structural ("box clues"), not yet a real blur/detail grid
  — a soft body paragraph with no literal box and no anchors nearby may be
  missed or, if picked, only reachable by ORB.
- The live target box uses a 2-point **similarity** transform (translation +
  scale + rotation), not a full homography, so under strong perspective it's
  approximate — the actual `composite` registration is a full homography.
- No exposure/white-balance matching between base and close-ups; visible seams
  are possible.
- Sticker / markers-only pages: close-ups fall to ORB-only (their fiducials have
  no known canvas coordinates).

## Deferred

- Native app for a real single high-res capture.
- Sub-registration for the corner-sticker and markers-only cases (their anchors
  have no known canvas coordinates, so close-ups there fall to ORB-only).
- Multi-frame super-resolution / stacking.
