# WJM MobileDeviceDemo

A phone-first web app that watches the camera, **captures the page the instant
its four corner markers sit inside the on-screen guide**, and **extracts its
structure** — page metadata, bullets, tags, references, contacts — as a JSON
object with a position for every element. All of it runs on-device in
WebAssembly; nothing is uploaded.

It reads the markers a WJM sheet is printed with (`wingjournal make-sheet` →
`DICT_4X4_50`, IDs `0/1/2/3` = TL/TR/BR/BL) **or** four identical adhesive
corner stickers (`wingjournal make-stickers`, ID `10`) — with stickers it also
guesses the physical page size. It mirrors the CLI's extraction stage for stage,
so a capture here matches `wingjournal ingest --parse-body`.

## Try it

**<https://wingtangwong.github.io/WingJournalMarkup/MobileDeviceDemo/>** — open
it on a phone, tap **Start camera**, hold a printed WJM sheet in the frame.

`vendor/opencv.js` and `vendor/tesseract/` (the WASM engines, ~17 MB) are
committed, so the whole demo serves from the repo / GitHub Pages with no
external calls.

## Run locally

```bash
cd MobileDeviceDemo
python3 serve.py           # http://localhost:8000
```

Open it in a desktop browser and hold a printed WJM sheet (or
`samples/wjm-writing-sheet-letter.png` on another screen) up to the webcam.
`./fetch-opencv.sh [version]` refreshes the vendored OpenCV.js.

### Serving it yourself over the LAN

The hosted link is the easy path. To serve your own copy to a phone,
`getUserMedia` needs a secure context — `localhost` over HTTP is fine, but a
phone reaching this machine over the LAN needs **HTTPS**:

```bash
# one throwaway cert (replace the IP with this machine's LAN address)
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
        -days 365 -subj "/CN=wjm-demo" -addext "subjectAltName=IP:192.168.1.50"

python3 serve.py --host 0.0.0.0 --https --cert cert.pem --key key.pem
# then browse to https://192.168.1.50:8000 on the phone and accept the warning
```

A tunnel (`cloudflared tunnel --url http://localhost:8000`, `ngrok http 8000`)
also gives you HTTPS without certs.

## How it works

| Where | What |
|---|---|
| **`js/worker.js`** (Web Worker) | OpenCV.js (WASM). Per-frame ArUco detection; `analyze` picks the sharpest burst frame (Tenengrad), rectifies to a fixed 2550×3300 canvas, runs the recognition geometry, plans close-up targets, and keeps the canvas in a session; `composite` registers + feather-pastes each close-up (anchor / ORB homography); `finish` re-recognises the composite. Keeps the 11 MB compile and all pixel work off the UI thread. |
| **Tesseract.js** (its own worker, spawned from the main thread) | OCR of the metadata cells and each body line off the rectified page. |
| **`js/app.js`** (main thread) | Camera, guide overlay, auto-capture trigger, and the recognition tail: crop each region → OCR → parse. |
| **`js/wjm-parse.js`** | The WJM grammar + element parser (bullets, tags, temporal, references, contacts). Pure text, no dependencies. |

### The capture flow

See [`SCANNER.md`](SCANNER.md) for the full state machine. **Phases A + B**:

1. On start, the track is pushed to the sensor's max resolution + continuous
   autofocus (`applyConstraints`; Safari/iOS has no `ImageCapture.takePhoto()`).
2. Camera → hidden ≤900 px canvas → worker ~14×/s → `aruco_ArucoDetector`
   (`DICT_4X4_50`). Three-plus id-`10` markers → **sticker mode**.
3. Guide test: every corner marker inside the guide, large enough. Brackets go
   **red → orange → yellow → green** as the lock firms.
4. Four corner ids in the guide, stable 450 ms → **base burst** of 6 frames.
   No sharpness gate. The worker keeps the sharpest (Tenengrad), rectifies it
   into a fixed 2550×3300 canvas (INTER_AREA/LINEAR, never CUBIC), and **plans
   up to 5 close-up targets** (metadata block, literal regions, text-line
   clusters).
5. **Close-up pass** (auto only): for each target the overlay shows a dashed box
   positioned from the visible markers — *keep two markers in view, move
   closer*. On a steady framing it bursts again; the worker registers the
   close-up (`anchorHomography` from ≥2 shared ArUco ids, `orbHomography`
   fallback) and feather-composites it into the canvas. **Skip this** /
   **Use what I have** end the pass early.
6. `finish` recognises the composited canvas; the main thread OCRs each cell and
   line, runs `wjm-parse`, assembles the record. The review screen lists which
   close-ups composited and how.

**Not yet built (phase C):** a re-assess loop that re-plans targets on the
composite and offers another pass.

### The captured report

Capture opens a full-screen scrollable report (tap **Retake** to go back to the
camera). It's built for diagnosing OCR — screenshot or print it and send it
back:

- the **photo** and the **rectified page with every OpenCV-detected region
  outlined** (green = metadata block / cells / text lines, blue = literal image
  region);
- the parsed **extraction** — markers, sizes, how the block was found
  (registration marks / ruled lines), a **sharpness score with a per-fiducial
  acutance table** (soft probes flagged red), and all seven metadata fields;
- **Metadata cells — crops fed to OCR**: for each of `document_id`, `page_id`,
  `topic_tags`, `left`, `above`, `below`, `right`, the exact padded crop handed
  to Tesseract and what it read (or *nothing read*, in red);
- **Page body**: each segmented text line as its crop + OCR text + the element
  it parsed to; literal regions shown as *as-is image*.

Three downloads: the full-res **photo** (JPEG), the **rectified page** (JPEG),
and the **capture data** (JSON — crops omitted to keep it small):

```jsonc
{
  "dictionary": "DICT_4X4_50",
  "source": { "width": 4032, "height": 3024 },
  "page_frame_quad": [[x,y], …],            // outer marker corners, TL,TR,BR,BL
  "normalized": { "width": 2139, "height": 2800, "target_long_px": 2800 },
  "detected_fiducials": [ { "id":0, "role":"TOP_LEFT", "corners":[…], "center":[…] }, … ],
  "sharpness": { "score":0.71, "blurry":false, "rectified_score":0.55, "probes":[ … ] },
  "metadata_block": { "bbox":[…], "row1_cells":[…], "row2_cells":[…], "confidence":0.97 },
  "page_metadata": { "document_id":"Research", "page_id":"P017",
                     "topic_tags":["AI"], "left":null, "above":null,
                     "below":null, "right":null, "_confidence":0.57 },
  "literal_assets": [ { "bbox":[…], "confidence":… } ],
  "detected_elements": [
    { "kind":"heading",   "text":"Project Kickoff", "bbox":[x,y,w,h], "confidence":0.9, "data":{} },
    { "kind":"bullet",    "text":"* draft the spec", "bbox":[…], "data":{ "state":"open", "glyph":"*", "item":"draft the spec", "tags":[] } },
    { "kind":"tags",      "text":"#backend #api", "bbox":[…], "data":{ "tags":["backend","api"] } },
    { "kind":"reference", "text":"-> [Research:P017:AUTH]", "bbox":[…], "data":{ "document":"Research","page":"P017","anchor":"AUTH" } }
  ]
}
```

Every element carries a `bbox` in normalized-page coordinates, so a PDF/SVG can
be rebuilt from it (planned).

## Development workflow

**The CLI (`src/wingjournal/`) is the reference. This demo is a port of it.**
When an extraction stage changes there, port it here:

| CLI (`src/wingjournal/`) | Demo (`MobileDeviceDemo/js/`) |
|---|---|
| `vision/aruco.py`, `vision/boundary.py`, `vision/rectify.py` | `vision-core.js` |
| `recognition/metadata_block.py`, `recognition/text/segment.py`, `vision/literal_box.py` | `vision-core.js` |
| `recognition/tags.py`, `recognition/parse.py` | `wjm-parse.js` |
| `vision/corner_sticker.py` | `vision-core.js` |
| `recognition/text/*` (Tesseract) | `ocr.js` |
| `pipeline.ingest_image` | `worker.js` `onAnalyze` + `app.js` `runExtraction` |

`tests/parse.test.mjs` re-runs `tests/test_tags.py` + `tests/test_parse.py`
against `wjm-parse.js`; `tests/vision.test.mjs` smoke-tests the detectors against
the vendored OpenCV.js. Run: `node tests/parse.test.mjs && node tests/vision.test.mjs`.

## Requirements

- `getUserMedia` + WebAssembly. iOS Safari 14.5+ for capture; **SIMD WASM**
  (Safari 16.4+, Chrome 91+, Firefox 89+) for OCR — without it the demo still
  produces the full geometry, elements just come back with empty text.
- An OpenCV.js build with the ArUco bindings — `docs.opencv.org/4.x/opencv.js`
  (4.9.0+) has them; the app says so on start if a build doesn't.

## Files

```
index.html            layout + OPENCV_URLS
css/app.css            camera-viewfinder styling, safe-area aware
js/app.js              camera, overlay, auto-capture, OCR + parse tail (main thread)
js/worker.js           OpenCV.js (WASM); ArUco + geometric extraction (Web Worker)
js/vision-core.js      MarkerDetector, page quad, rectify, metadata block, segment, literals
js/wjm-parse.js        WJM grammar + element parser (pure text)
js/ocr.js              Tesseract.js wrapper + null fallback
serve.py               stdlib static server (HTTP, or HTTPS with --cert/--key)
fetch-opencv.sh        refresh vendor/opencv.js
vendor/                opencv.js + tesseract/ (committed WASM engines)
tests/                 parse.test.mjs, vision.test.mjs  (node)
```

## Limitations

Local demo — no batching, persistence, or upload; the JSON is yours to feed into
the CLI / store. Handwriting OCR is weak (Tesseract, same as the CLI's first
pass); machine-printed text reads well. Extraction quality tracks lighting and
how flat the page is held.
