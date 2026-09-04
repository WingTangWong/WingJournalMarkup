# WJM Marker Glyphs — WebDemoMarkers

A stand-alone test bench for a **different** fiducial from WJM's ArUco
corner markers: a hand-drawable **bullseye + 3-bit corner-code glyph**,
detected with plain OpenCV — no dictionary, no ML model. Independent of
`MobileDeviceDemo/` (different glyph, no shared state), but built the same
way: static site, no build step, OpenCV.js (WASM) in a Web Worker.

See `docs/ROADMAP.md`'s **D3 – Marker-glyph exploration** for how this fits
the rest of the project.

## The glyph

From the "Proposed Handwritten Syntax Lexicon" sketch — a redesign of spec
§11.4's hand-drawn ID-box notation. §11.4 used a filled square as the
always-present anchor corner; this swaps that for a **bullseye** (a ring
with a dot in its center), because a circle is far more tolerant of
hand-drawn wobble than a square corner, and — being a different *shape*
from the 3 code squares rather than just a different corner — it can't be
confused with one of them.

```
        ◎ ← bullseye (ring + center dot), a small gap outside the box
     ┌──────────────┐
     │ ▪            ▪│  ← upper-right corner: filled = 1, empty = 0
     │              │
     │▪            ▪ │  ← lower-right / lower-left, same rule
     └──────────────┘
```

- The **bullseye** sits just outside one corner of the box outline (with a
  gap — it's a separate shape, not touching), and by convention marks that
  corner as the box's "upper-left."
- The box's other **three corners** — upper-right, lower-right, lower-left,
  walking clockwise from the bullseye-marked corner — may each carry a
  small filled square just inside that corner, or be left plain.
- Those three corners are a **3-bit code**: `value = (upperRight << 2) |
  (lowerRight << 1) | lowerLeft`, so 0 (all empty, "None") through 7 (all
  filled).

Because the reading always starts from whichever corner is nearest the
bullseye and walks clockwise, the UR/LR/LL assignment stays correct no
matter how the glyph is rotated or tilted in frame — the bullseye (found
independently of the box's orientation) is what anchors the reading, not an
assumption about which way is "up" on screen.

## How detection works (`js/glyph-detect.js`)

Pure classical CV, no ArUco, no ML:

1. **Binarize**: `GaussianBlur` + Otsu threshold (ink = 255), same as
   `vision/registration.py`'s registration-mark detector.
2. **Find bullseyes**: `findContours(RETR_TREE)`, then for each top-level
   contour, check it's roughly circular (`4π·area/perimeter²`, tolerant of
   hand-drawn wobble), that it has a hole (the ring's white interior), and
   that the hole in turn has a child contour (the center dot) of plausible
   size, centered near the ring's center. This is the exact nested-contour
   technique the concentric-square registration marks use — circular
   instead of square.
3. **Find box quads**: `approxPolyDP` on the other top-level contours,
   keeping 4-point convex results with a filled-enough bounding box.
4. **Pair** each bullseye with its nearest plausibly-sized quad (nearest
   distance wins first, so two glyphs near each other don't steal one
   another's box).
5. **Orient**: sort the quad's corners clockwise on screen, find the one
   nearest the bullseye (the anchor), and read the next three clockwise as
   upper-right / lower-right / lower-left.
6. **Decode**: probe a small square just inside each of those 3 corners
   (inset toward the box's center) for ink density; > ~38% dark → bit = 1.

Runs per-frame in `js/worker.js` (a Web Worker, so the camera preview never
stutters) at a downscaled processing width (`PROC_W` in `js/app.js`), same
pattern as `MobileDeviceDemo`.

## Status

First pass, verified only against **synthetic** glyphs (drawn with OpenCV's
own primitives — see `tests/glyph-detect.test.mjs`): all 8 values decode
correctly, a 25°-rotated glyph still decodes correctly, and two glyphs in
one frame both decode without cross-pairing. **Not yet run against real
hand-drawn ink** — marker-pen wobble, uneven stroke width, and real
paper/lighting noise will likely need the same tolerance-tuning pass the
project's OCR/HTR work went through (the circularity/size/ink-density
thresholds in `js/glyph-detect.js` are commented as tuning knobs for
exactly that reason).

Out of scope for this pass: mapping the decoded 0–7 value back to WJM's
semantic metadata fields (document_id/page_id/left/above/below/right) —
this demo stops at detect → decode → visualize.

## Running it

```bash
./fetch-opencv.sh          # only needed if vendor/opencv.js is missing/stale
python3 serve.py           # http://localhost:8000  (camera OK on localhost)
python3 serve.py --host 0.0.0.0 --https --cert cert.pem --key key.pem   # phone on LAN
```

Point the camera at a printed or hand-drawn glyph; detected ones are
outlined in green with small dots at the 3 code corners (filled = 1, hollow
= 0) and the decoded value drawn on the box.

## Tests

```bash
node tests/glyph-detect.test.mjs        # needs vendor/opencv.js
```

## Files

| File | Role |
|---|---|
| `index.html` | camera view, overlay canvas, start gate |
| `js/app.js` | camera capture, worker messaging, overlay drawing (main thread) |
| `js/worker.js` | OpenCV.js load + per-frame detection protocol (Web Worker) |
| `js/glyph-detect.js` | the actual detection algorithm (classic script, shared by worker + tests) |
| `vendor/opencv.js` | vendored OpenCV.js (WASM); `fetch-opencv.sh` refreshes it |
| `serve.py` | local static server (HTTP/HTTPS) |
| `tests/glyph-detect.test.mjs` | synthetic-glyph correctness tests |
