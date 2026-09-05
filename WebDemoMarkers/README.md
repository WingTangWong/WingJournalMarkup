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
2. **Find bullseyes**: `cv.HoughCircles` on a median-blurred grayscale.
   Gradient voting finds a circle even when the ring has a gap or an uneven,
   retraced stroke — a contour-topology test can't. Each candidate is then
   confirmed against the ink mask by sampling three concentric bands: a dark
   center (the dot), a light middle, a dark outer ring.
3. **Group ink into components** once per frame, keeping only those long
   enough to plausibly be a box edge (a real edge runs a couple of bullseye
   diameters; an individual handwritten letter stroke does not). This — not
   distance — is what keeps nearby writing out of the fit.
4. **Fit the box**, seeded from the single qualifying component *nearest the
   bullseye* (by design the intended box sits right beside it, far closer
   than any neighbouring glyph's box), then grown: each round, `minAreaRect`
   refits over whatever qualifying ink landed near the current fit's
   boundary. A gappy box's other sides join as the fit reaches them; a
   neighbouring glyph's box never does.
5. **Orient**: sort the fitted rectangle's corners clockwise on screen, find
   the one nearest the bullseye (the anchor), and read the next three
   clockwise as upper-right / lower-right / lower-left.
6. **Decode**: probe just inside each of those 3 corners for ink density —
   a low bar, so a hatched or scribbled mark counts, not only a solid fill.

Nothing here requires the box to be one connected shape, a closed loop, or
made of straight lines, which is what makes it tolerant of real hand-drawn
error. Runs per-frame in `js/worker.js` (a Web Worker, so the camera preview
never stutters) at a downscaled processing width (`PROC_W` in `js/app.js`),
same pattern as `MobileDeviceDemo`.

## Status

**Synthetic** (`tests/glyph-detect.test.mjs`, all 14 passing): every 3-bit
value, open-bracket and closed boxes, 25° rotation, mid-line gaps, wobbly
non-parallel edges, a hatched rather than solid corner mark, stray
handwriting-like ink beside the glyph, two glyphs at real-sheet packing
density, plus negative cases (blank page; a plain rectangle with no
bullseye).

**Real photo** — a hand-drawn 4x4 reference sheet (16 glyphs, packed tightly,
with label text close enough above the first glyph to have wrecked an earlier
version's fit):

| | result |
|---|---|
| glyphs located | **8 of 16**, no false positives |
| box geometry | accurate — the fitted boxes land on the real glyphs |
| values decoded | **roughly half right** by eye; the rest still wrong |
| speed | ~80ms at the demo's live 900px processing width |

The two earlier versions of this detector found *nothing* on that sheet, so
this is a large step — but **located is not decoded**, and neither number is
finished work. That sheet is also a deliberately hard case: 16 glyphs at a
density a real page would rarely have.

Diagnosing against that real photo is what drove the design. Two findings
were worth more than any amount of threshold tuning: the boxes are drawn as
**open 3-sided brackets** (the side facing the bullseye is simply not drawn),
and a hand-drawn ring's traced contour is often nowhere near circular in the
strict `4π·area/perimeter²` sense and can have a gap that breaks contour
nesting outright. Both broke the original approach at a level no parameter
could fix.

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
