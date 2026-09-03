# WJM MobileDeviceDemo

A phone-first web app that watches the camera and **captures the page the
instant its four corner markers sit inside the on-screen guide**. All computer
vision runs on-device in WebAssembly (OpenCV.js) — nothing is uploaded.

It captures the same markers a WJM sheet is printed with
(`wingjournal make-sheet` → `DICT_4X4_50`, IDs `0/1/2/3` = TL/TR/BR/BL), so a
capture here lines up with `wingjournal ingest`.

## Try it

**<https://wingtangwong.github.io/WingJournalMarkup/MobileDeviceDemo/>** — open
it on a phone, tap **Start camera**, hold a printed WJM sheet in the frame.

`vendor/opencv.js` (the ~11 MB WASM engine) is committed, so the whole demo
serves from the repo / GitHub Pages with no external dependency.

## Run locally

```bash
cd MobileDeviceDemo
python3 serve.py           # http://localhost:8000
```

Open `http://localhost:8000` in a desktop browser and hold a printed WJM sheet
(or the on-screen `samples/wjm-writing-sheet-letter.png`) up to the webcam.
`./fetch-opencv.sh [version]` refreshes the vendored OpenCV.js.

### Serving it yourself over the LAN

The hosted link above is the easy path. To serve your own copy to a phone,
`getUserMedia` needs a secure context — `localhost` over HTTP is fine, but a
phone reaching this machine over the LAN needs **HTTPS**:

```bash
# one throwaway cert (replace the IP with this machine's LAN address)
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
        -days 365 -subj "/CN=wjm-demo" -addext "subjectAltName=IP:192.168.1.50"

python3 serve.py --host 0.0.0.0 --https --cert cert.pem --key key.pem
# then browse to https://192.168.1.50:8000 on the phone and accept the warning
```

A tunnel (`cloudflared tunnel --url http://localhost:8000`, `ngrok http 8000`,
etc.) also gives you HTTPS without certs.

## How it works

OpenCV.js (11 MB WASM) and every ArUco detection run in **`js/worker.js`**, a
Web Worker — so the 11 MB compile and the per-frame work never stall the camera
preview or the overlay. The main thread only grabs frames, draws the overlay,
and shows results.

| Stage | |
|---|---|
| Camera | `getUserMedia` rear camera → a hidden ≤900 px canvas; `getImageData` buffer transferred to the worker ~14×/s |
| Detect | worker: OpenCV.js `aruco_ArucoDetector` (`DICT_4X4_50`) with sub-pixel corners |
| Guide test | main thread: every marker's four corners must fall inside the guide rectangle, and be large enough (rejects "too far") |
| Trigger | all of ids 0/1/2/3 inside & stable for 450 ms → auto-shutter (toggle **Auto** off for manual only) |
| Output | full-res JPEG **+** a perspective-rectified PNG (worker: `getPerspectiveTransform` / `warpPerspective`, longer side 1600 px, aspect clamped to `[1.15, 1.6]` — same as `wingjournal`) **+** a JSON sidecar (marker ids, corners, page-frame quad) |

The overlay maps between three coordinate spaces — the processing canvas, the
intrinsic camera frame, and the `object-fit: cover` on-screen video — so the
marker outlines and the guide line up regardless of aspect ratio.

`js/vision-core.js` (loaded into the worker) mirrors
`wingjournal.vision.{aruco,boundary,rectify}`: role-by-id, page quad from the
*outer* corner of each marker, `output_size`.

## Requirements

- A browser with `getUserMedia` + WebAssembly (iOS Safari 14.5+, Chrome/Firefox/Edge).
- An OpenCV.js build with the ArUco bindings — the default
  `https://docs.opencv.org/4.x/opencv.js` (4.9.0+) has them. If yours doesn't,
  the app says so on start; vendor a different build to `vendor/opencv.js`.

## Files

```
index.html            layout + the OPENCV_URLS list
css/app.css            camera-viewfinder styling, safe-area aware
js/app.js              camera, frame pump, overlay, auto-capture, results (main thread)
js/worker.js           loads OpenCV.js (WASM); runs detection + rectification
js/vision-core.js      MarkerDetector, pageQuadFromMarkers, rectify (worker side)
serve.py               stdlib static server (HTTP, or HTTPS with --cert/--key)
fetch-opencv.sh        vendor OpenCV.js into vendor/ (git-ignored)
```

## Limitations

Local demo only — no batching, no persistence, no upload. A capture's JSON is
downloadable but you feed it to the CLI/store yourself. Detection quality tracks
lighting and how flat the page is held, same as the pipeline.
