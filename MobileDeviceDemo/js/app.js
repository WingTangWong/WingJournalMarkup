// WJM MobileDeviceDemo — auto-capture a page the moment its four corner markers
// sit inside the on-screen guide, then extract its structure.
//
// The camera preview, overlay and UI run here on the main thread. OpenCV.js
// (WASM) + ArUco + the geometric extraction run in js/worker.js; Tesseract.js
// OCR runs from here (it manages its own worker). js/wjm-parse.js (loaded as a
// classic script) turns recognized text into typed elements.

import { getRecognizer } from "./ocr.js";

const OPENCV_URLS = (window.OPENCV_URLS || ["vendor/opencv.js"]).map(
  (u) => new URL(u, location.href).href,
);
const TESSERACT_PATHS = {
  tesseractJs: new URL("vendor/tesseract/tesseract.min.js", location.href).href,
  workerPath: new URL("vendor/tesseract/worker.min.js", location.href).href,
  corePath: new URL("vendor/tesseract/tesseract-core-simd-lstm.wasm.js", location.href).href,
  langPath: new URL("vendor/tesseract/", location.href).href,
};
const ROLE_ORDER = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"];

// ---- tunables ----------------------------------------------------------
const PROC_W = 900;           // detection runs on a frame this wide
const TICK_MS = 70;           // min gap between detection requests
const HOLD_MS = 450;          // all-4-inside must persist this long to fire
const STALE_MS = 260;         // ignore marker results older than this for the trigger
const EDGE_SLACK = 0.012;     // fraction of guide width tolerated outside the box
const MIN_MARKER_FRAC = 0.018; // reject markers smaller than this * guide width
const LIVE_MIN_SHARPNESS = 0.32; // auto-shutter blocked below this frame score
const JPEG_QUALITY = 0.92;
const HOLD_GREEN_FRAC = 0.55;    // ring fill past which the guide turns green
const REJECT_COOLDOWN_MS = 1100; // after a blurry grab, coach for this long before re-arming

// red → orange → yellow → green as the lock firms up
const PHASE_COLOR = { seek: "#e5533d", frame: "#e8873a", hold: "#f2c14e", sharp: "#2ecc71" };

// ---- dom --------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const statusEl = $("status");
const pips = Object.fromEntries(
  [...document.querySelectorAll(".pip")].map((el) => [el.dataset.role, el]),
);

// ---- state ----------------------------------------------------------
let worker = null;
let stream = null;
let track = null;
let facing = "environment";
let torchOn = false;
let running = false;
let busy = false;            // a capture is being processed
let pending = false;         // a detect request is in flight
let seq = 0;
let lastSent = 0;
let lockStart = 0;
let markers = [];            // latest markers, in intrinsic (full-res) pixels
let markersAt = 0;
let frameSharp = null;       // {score, blurry} for the latest frame
let fiducialMode = "sheet";  // "sheet" | "stickers"
let cooldownUntil = 0;       // re-arm auto-capture only after this (post-reject coaching)
let coachMsg = "";           // what the last blurry grab told the user to fix
let softRoles = new Set();   // fiducials the last grab found soft, for the overlay
let recognizer = null;
let ocrBackend = "none";
const analyzeWaiters = new Map();

const procCanvas = document.createElement("canvas");
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });
const grabCanvas = document.createElement("canvas");
const rectCanvas = document.createElement("canvas");
const work = document.createElement("canvas"); // per-region OCR scratch
const objectUrls = [];

// ---- boot ---------------------------------------------------------
(function boot() {
  setStatus("Loading the vision engine…");
  worker = new Worker("js/worker.js");
  worker.onerror = (e) => fatal(`worker failed: ${e.message || e}`);
  worker.onmessage = onWorkerMessage;
  worker.postMessage({ type: "init", opencvUrls: OPENCV_URLS });

  // warm up OCR in parallel; falls back to a null recognizer if it can't load
  getRecognizer("auto", TESSERACT_PATHS).then((rec) => {
    recognizer = rec;
    ocrBackend = rec.name;
  }).catch(() => { ocrBackend = "none"; });
})();

function onWorkerMessage(e) {
  const m = e.data;
  switch (m.type) {
    case "ready": {
      const btn = $("btn-start");
      btn.disabled = false;
      btn.textContent = "Start camera";
      btn.addEventListener("click", start);
      setStatus("Ready — tap “Start camera”.");
      break;
    }
    case "error":
      fatal(m.message);
      break;
    case "markers": {
      pending = false;
      const k = procCanvas.width / (video.videoWidth || procCanvas.width) || 1;
      markers = m.markers.map((mk) => ({
        id: mk.id,
        role: mk.role,
        corners: mk.corners.map(([x, y]) => [x / k, y / k]),
        center: [mk.center[0] / k, mk.center[1] / k],
      }));
      markersAt = performance.now();
      frameSharp = m.sharpness || null;
      fiducialMode = m.mode || "sheet";
      break;
    }
    case "progress":
      if (busy) setStatus(`Analyzing — ${m.stage}…`, "ok");
      break;
    case "analyzed": {
      const w = analyzeWaiters.get(m.seq);
      if (w) { analyzeWaiters.delete(m.seq); w(m); }
      break;
    }
  }
}

// ---- camera -----------------------------------------------------
async function start() {
  try {
    $("btn-start").disabled = true;
    stopStream();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facing },
        width: { ideal: 3840 },
        height: { ideal: 2160 },
      },
    });
    track = stream.getVideoTracks()[0];
    video.srcObject = stream;
    await video.play().catch(() => {});
    // a canvas/stream source can reach metadata before we could attach a listener
    if (video.readyState < 1) await once(video, "loadedmetadata");
    await maximizeResolution();

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    procCanvas.width = Math.min(PROC_W, vw);
    procCanvas.height = Math.round(procCanvas.width * (vh / vw));

    setupTorchButton();
    $("gate").hidden = true;
    $("result").hidden = true;
    running = true;
    lockStart = 0;
    markers = [];
    cooldownUntil = 0;
    coachMsg = "";
    softRoles = new Set();
    requestAnimationFrame(loop);
  } catch (err) {
    $("btn-start").disabled = false;
    if (err && err.name === "NotAllowedError") {
      $("gate-fine").textContent =
        "Camera permission was denied. Enable it for this site and reload.";
    } else {
      fatal(`Camera error: ${err.message || err}`);
    }
  }
}

function stopStream() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = track = null;
  torchOn = false;
}

function setupTorchButton() {
  const btn = $("btn-torch");
  const caps = track && track.getCapabilities ? track.getCapabilities() : {};
  if (caps && caps.torch) {
    btn.hidden = false;
    btn.onclick = async () => {
      torchOn = !torchOn;
      btn.setAttribute("aria-pressed", String(torchOn));
      try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); }
      catch { /* ignore */ }
    };
  } else {
    btn.hidden = true;
  }
}

// Push the live track to the sensor's best resolution + continuous autofocus.
// Safari/iOS has no ImageCapture.takePhoto(), so the "sharp photo" is just the
// best frame this stream can give — ask for as much as the hardware allows.
async function maximizeResolution() {
  if (!track || !track.getCapabilities) return;
  let caps;
  try { caps = track.getCapabilities(); } catch { return; }
  const want = {};
  // cap at ~4K: past that the stream often can't hold framerate and some
  // phones just fail the constraint outright
  if (caps.width && caps.width.max) want.width = { ideal: Math.min(caps.width.max, 3840) };
  if (caps.height && caps.height.max) want.height = { ideal: Math.min(caps.height.max, 2160) };
  if (Array.isArray(caps.focusMode) && caps.focusMode.includes("continuous")) {
    want.focusMode = "continuous";
  }
  if (Array.isArray(caps.exposureMode) && caps.exposureMode.includes("continuous")) {
    want.exposureMode = "continuous";
  }
  if (!Object.keys(want).length) return;
  try {
    await track.applyConstraints(want);
    // give the sensor a beat to actually switch mode / refocus
    await new Promise((r) => setTimeout(r, 250));
  } catch { /* keep whatever getUserMedia gave us */ }
}

// ---- main loop --------------------------------------------------
function loop(ts) {
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;

  // hand a fresh frame to the worker when it is idle
  if (!pending && !busy && ts - lastSent >= TICK_MS) {
    procCtx.drawImage(video, 0, 0, procCanvas.width, procCanvas.height);
    const img = procCtx.getImageData(0, 0, procCanvas.width, procCanvas.height);
    pending = true;
    lastSent = ts;
    seq += 1;
    worker.postMessage(
      { type: "detect", seq, width: img.width, height: img.height, buffer: img.data.buffer },
      [img.data.buffer],
    );
  }

  const view = computeView();
  const guide = computeGuide(view.cssW, view.cssH);
  const gi = screenRectToIntrinsic(guide, view);

  const assessed = markers.map((m) => assessMarker(m, gi));
  const goodRoles = new Set(
    assessed.filter((a) => a.inside && a.bigEnough && a.marker.role).map((a) => a.marker.role),
  );
  const allIn = ROLE_ORDER.every((r) => goodRoles.has(r));
  const fresh = performance.now() - markersAt < STALE_MS;
  // live gate is lenient (raw-frame lapvar runs low); the report re-scores the
  // rectified page against the stricter threshold
  const blurry = !!(frameSharp && frameSharp.score < LIVE_MIN_SHARPNESS);

  const cooling = performance.now() < cooldownUntil;
  if (!cooling && coachMsg) { coachMsg = ""; softRoles = new Set(); }

  // lock firms up as: corners seen (orange) → all inside & steady (yellow) →
  // held long enough on a sharp frame (green) → grab + verify the real photo
  const armed = allIn && fresh && !blurry && !cooling;
  if (armed && $("chk-auto").checked && !busy) {
    if (!lockStart) lockStart = ts;
  } else {
    lockStart = 0;
  }
  const held = lockStart ? ts - lockStart : 0;
  setLockRing(lockStart ? Math.min(1, held / HOLD_MS) : 0);

  let phase;
  if (goodRoles.size < 4) phase = assessed.length ? "frame" : "seek";
  else if (blurry || cooling || !lockStart) phase = "hold";
  else phase = held >= HOLD_MS * HOLD_GREEN_FRAC ? "sharp" : "hold";

  updatePips(goodRoles);
  drawOverlay(assessed, guide, view, phase);
  updateStatus(assessed, goodRoles, blurry, cooling);

  if (lockStart && held >= HOLD_MS && !busy) capture("auto");
}

// ---- geometry -------------------------------------------------
// object-fit: cover — which intrinsic-pixel rect is actually on screen
function computeView() {
  const r = video.getBoundingClientRect();
  const vw = video.videoWidth || r.width || 1;
  const vh = video.videoHeight || r.height || 1;
  const s = Math.max(r.width / vw, r.height / vh);
  return {
    cssW: r.width, cssH: r.height, s,
    visX: (vw - r.width / s) / 2,
    visY: (vh - r.height / s) / 2,
  };
}

function computeGuide(cssW, cssH) {
  const top = 74, bottom = 112, side = 16;
  const availW = cssW - side * 2;
  const availH = cssH - top - bottom;
  const aspect = 11 / 8.5; // WJM Letter sheet, portrait
  let gw = availW;
  let gh = gw * aspect;
  if (gh > availH) { gh = availH; gw = gh / aspect; }
  return { x: (cssW - gw) / 2, y: top + Math.max(0, (availH - gh) / 2), w: gw, h: gh };
}

const screenRectToIntrinsic = (g, v) => ({
  x: v.visX + g.x / v.s, y: v.visY + g.y / v.s, w: g.w / v.s, h: g.h / v.s,
});
const intrinsicToScreen = ([x, y], v) => [(x - v.visX) * v.s, (y - v.visY) * v.s];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function assessMarker(m, gi) {
  const slack = gi.w * EDGE_SLACK;
  const inside = m.corners.every(
    ([x, y]) =>
      x >= gi.x - slack && x <= gi.x + gi.w + slack &&
      y >= gi.y - slack && y <= gi.y + gi.h + slack,
  );
  const side =
    (dist(m.corners[0], m.corners[1]) + dist(m.corners[1], m.corners[2]) +
     dist(m.corners[2], m.corners[3]) + dist(m.corners[3], m.corners[0])) / 4;
  return { marker: m, inside, bigEnough: side >= gi.w * MIN_MARKER_FRAC };
}

// ---- rendering -----------------------------------------------
function drawOverlay(assessed, guide, view, phase) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(view.cssW * dpr);
  const H = Math.round(view.cssH * dpr);
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, view.cssW, view.cssH);

  // dim everything outside the guide
  octx.fillStyle = "rgba(0,0,0,0.34)";
  octx.beginPath();
  octx.rect(0, 0, view.cssW, view.cssH);
  octx.rect(guide.x, guide.y, guide.w, guide.h);
  octx.fill("evenodd");

  // corner brackets — colour tracks the lock phase
  const col = PHASE_COLOR[phase] || PHASE_COLOR.hold;
  const L = Math.min(guide.w, guide.h) * 0.12;
  octx.strokeStyle = col;
  octx.lineWidth = 4;
  octx.lineCap = "round";
  const bracket = (cx, cy, dx, dy) => {
    octx.beginPath();
    octx.moveTo(cx + dx * L, cy);
    octx.lineTo(cx, cy);
    octx.lineTo(cx, cy + dy * L);
    octx.stroke();
  };
  bracket(guide.x, guide.y, 1, 1);
  bracket(guide.x + guide.w, guide.y, -1, 1);
  bracket(guide.x + guide.w, guide.y + guide.h, -1, -1);
  bracket(guide.x, guide.y + guide.h, 1, -1);

  for (const a of assessed) {
    const pts = a.marker.corners.map((p) => intrinsicToScreen(p, view));
    octx.beginPath();
    pts.forEach(([x, y], i) => (i ? octx.lineTo(x, y) : octx.moveTo(x, y)));
    octx.closePath();
    const soft = a.marker.role && softRoles.has(a.marker.role);
    const ok = a.inside && a.bigEnough && a.marker.role !== null && !soft;
    octx.strokeStyle = ok ? "#2ecc71" : "#e5533d";
    octx.fillStyle = ok ? "rgba(46,204,113,0.22)" : "rgba(229,83,61,0.16)";
    octx.lineWidth = 3;
    octx.fill();
    octx.stroke();

    const [lx, ly] = intrinsicToScreen(a.marker.center, view);
    octx.fillStyle = "#fff";
    octx.font = "700 12px -apple-system, system-ui, sans-serif";
    octx.textAlign = "center";
    const tag = a.marker.role ? a.marker.role.replace("_", " ") : `#${a.marker.id}`;
    octx.fillText(soft ? `${tag} · soft` : tag, lx, ly + 4);
  }
}

function updatePips(goodRoles) {
  for (const [role, el] of Object.entries(pips)) el.classList.toggle("on", goodRoles.has(role));
}

function updateStatus(assessed, goodRoles, blurry, cooling) {
  const n = assessed.length;
  if (cooling && coachMsg) {
    setStatus(coachMsg, "warn");
  } else if (goodRoles.size === 4 && blurry) {
    setStatus("Too blurry — hold steady", "warn");
  } else if (goodRoles.size === 4) {
    setStatus("Hold still…", "ok");
  } else if (n === 0) {
    setStatus("Point at a WJM sheet");
  } else {
    const outside = assessed.some((a) => a.marker.role && !a.inside);
    const small = assessed.some((a) => a.marker.role && a.inside && !a.bigEnough);
    setStatus(
      small ? "Move closer" :
      outside ? "Fit the whole sheet in the frame" :
      `Found ${goodRoles.size}/4 corners`,
      "warn",
    );
  }
}

const setLockRing = (v) =>
  document.documentElement.style.setProperty("--lock", v.toFixed(3));

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${cls}`;
}

// ---- capture ------------------------------------------------
async function capture(trigger) {
  if (busy) return;
  busy = true;
  running = false;
  setLockRing(0);

  navigator.vibrate?.(trigger === "auto" ? [35, 25, 35] : 30);
  const flash = $("flash");
  flash.classList.remove("fire");
  void flash.offsetWidth;
  flash.classList.add("fire");

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  grabCanvas.width = vw;
  grabCanvas.height = vh;
  const gctx = grabCanvas.getContext("2d");
  gctx.drawImage(video, 0, 0, vw, vh);

  const capturedMarkers = markers; // already intrinsic (full-res) pixels
  const stamp = new Date();
  const gate = trigger === "auto"; // manual shutter always produces a result

  setStatus(gate ? "Checking sharpness…" : "Analyzing…", "ok");
  const full = gctx.getImageData(0, 0, vw, vh);
  const seqId = ++seq;
  const msg = await new Promise((resolve) => {
    analyzeWaiters.set(seqId, resolve);
    worker.postMessage(
      { type: "analyze", seq: seqId, width: vw, height: vh, buffer: full.data.buffer, markers: capturedMarkers, mode: fiducialMode, gate },
      [full.data.buffer],
    );
    setTimeout(() => {
      if (analyzeWaiters.has(seqId)) { analyzeWaiters.delete(seqId); resolve({ error: "analysis timed out" }); }
    }, 20000);
  });

  // the real photo came back soft — coach and drop straight back to the live
  // feed to re-focus and re-lock (spec §9.1)
  if (msg && msg.rejected) {
    const roles = (msg.soft || [])
      .map((n) => { const m = /(\d+)$/.exec(n); return m ? ROLE_ORDER[+m[1]] : null; })
      .filter(Boolean);
    softRoles = new Set(roles);
    coachMsg = roles.length
      ? `Soft at ${roles.map((r) => r.replace("_", " ").toLowerCase()).join(", ")} — move closer & hold steady`
      : "Too blurry — move closer, steady the phone, let it focus";
    cooldownUntil = performance.now() + REJECT_COOLDOWN_MS;
    navigator.vibrate?.(120);
    setStatus(coachMsg, "warn");
    lockStart = 0;
    setLockRing(0);
    busy = false;
    running = true;
    requestAnimationFrame(loop);
    return;
  }

  const rawUrl = await canvasUrl(grabCanvas, "image/jpeg", JPEG_QUALITY);
  let capture;
  let rectUrl = null;

  if (msg && msg.geometry && msg.rectified) {
    const r = msg.rectified;
    rectCanvas.width = r.width;
    rectCanvas.height = r.height;
    rectCanvas.getContext("2d").putImageData(
      new ImageData(new Uint8ClampedArray(r.buffer), r.width, r.height), 0, 0,
    );
    // JPEG: the rectified page can now be ~2800 px on the long side and a PNG
    // encode of that stalls the main thread; OCR crops come off rectCanvas
    // (the raw pixels), not this URL, so quality there is unaffected
    rectUrl = await canvasUrl(rectCanvas, "image/jpeg", JPEG_QUALITY);
    capture = await runExtraction(msg.geometry, rectCanvas);
  } else {
    capture = {
      app: "WJM MobileDeviceDemo",
      error: (msg && msg.error) || "analysis failed",
      source: { width: vw, height: vh },
      detected_fiducials: capturedMarkers.map((m) => ({
        id: m.id, role: m.role, corners: m.corners.map(([x, y]) => [round1(x), round1(y)]),
      })),
    };
  }
  capture.app = "WJM MobileDeviceDemo";
  capture.captured_at = stamp.toISOString();
  capture.trigger = trigger;
  capture.engine = `OpenCV.js${ocrBackend === "tesseract" ? " + Tesseract.js" : ""} (WebAssembly, Web Worker)`;

  // crop data URLs are for the on-screen report only — keep the JSON lean
  const jsonUrl = URL.createObjectURL(new Blob(
    [JSON.stringify(capture, (k, v) => (k === "cropUrl" ? undefined : v), 2)],
    { type: "application/json" },
  ));
  objectUrls.push(rawUrl, jsonUrl);
  if (rectUrl) objectUrls.push(rectUrl);
  showResult({ rawUrl, rectUrl, jsonUrl, capture, stamp, error: capture.error });
  setStatus("Captured", "ok");
  busy = false;
}

const METADATA_FIELDS = ["document_id", "page_id", "topic_tags", "left", "above", "below", "right"];
const mean = (xs) => (xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : 0);
const round3 = (v) => Math.round(v * 1000) / 1000;

// crops are kept as data URLs for the on-screen report only (not the JSON)
const cropUrls = [];

// OCR the metadata cells + body lines off the rectified page, then parse — the
// main-thread mirror of ingest_image's recognition tail (metadata.py / parse.py).
// Also records every crop it feeds to OCR, for the debug report.
async function runExtraction(geometry, rect) {
  const g = geometry;
  const P = window.WJMParse;
  const canOcr = recognizer && recognizer.name !== "none";

  const capture = {
    dictionary: g.dictionary,
    fiducial_mode: g.fiducial_mode || "printed_sheet",
    source: g.source,
    page_frame_quad: g.page_frame_quad,
    orientation: g.orientation,
    normalized: g.normalized,
    page_size_estimate: g.page_size_estimate || null,
    detected_fiducials: g.detected_fiducials,
    metadata_block: g.metadata_block,
    sharpness: g.sharpness || null,
    page_metadata: null,
    text_backend: recognizer ? recognizer.name : "none",
    literal_assets: g.literal_assets,
    detected_elements: [],
    line_boxes: g.line_boxes,
    _debug: { metadata_cells: [], body: [] },
    notes: [],
  };

  // crop a region of the rectified page, pad it white (what OCR sees), keep the
  // image, and — unless doOcr is false — recognize it.
  const region = async (bbox, doOcr = true) => {
    if (!bbox) return { text: "", words: [], recognized: false, cropUrl: null };
    const [x, y, w, h] = bbox.map((v) => Math.round(v));
    if (w < 1 || h < 1) return { text: "", words: [], recognized: false, cropUrl: null };
    const pad = 10;
    work.width = w + pad * 2;
    work.height = h + pad * 2;
    const wc = work.getContext("2d");
    wc.fillStyle = "#fff";
    wc.fillRect(0, 0, work.width, work.height);
    wc.drawImage(rect, x, y, w, h, pad, pad, w, h);
    const cropUrl = capCropUrl(work);
    cropUrls.push(cropUrl);
    if (!canOcr || !doOcr) return { text: "", words: [], recognized: false, cropUrl };
    const r = await recognizer.recognize(work);
    return Object.assign({ cropUrl }, r);
  };

  // ---- metadata cells (always report all 7 fields, in canonical order) ----
  setStatus("Reading the header…", "ok");
  const padCells = (cells, n) => (cells || []).concat(Array(n).fill(null)).slice(0, n);
  const cells = g.metadata_block
    ? [...padCells(g.metadata_block.row1_cells, 3), ...padCells(g.metadata_block.row2_cells, 4)]
    : Array(7).fill(null);
  const rowText = [];
  const cellConf = [];
  for (let i = 0; i < 7; i++) {
    const r = await region(cells[i]);
    let text = "";
    let confidence = 0;
    if (r.recognized) {
      const cs = r.words.map((w) => w.confidence);
      confidence = round3(mean(cs));
      for (const c of cs) cellConf.push(c);
      text = r.text;
    }
    rowText.push(text);
    capture._debug.metadata_cells.push({
      field: METADATA_FIELDS[i], bbox: cells[i], text, confidence,
      recognized: !!r.recognized, cropUrl: r.cropUrl,
    });
  }
  capture.page_metadata = Object.assign(
    {}, P.parseMetadataCells(rowText.slice(0, 3), rowText.slice(3)),
    { _confidence: round3(mean(cellConf)) },
  );

  // ---- body lines ----
  setStatus("Reading the body…", "ok");
  const lines = [];
  for (const box of g.line_boxes) {
    const r = await region(box);
    const cs = (r.words || []).map((w) => w.confidence);
    lines.push({
      text: r.recognized ? r.text : "",
      bbox: box, confidence: round3(mean(cs)), recognized: !!r.recognized,
    });
    capture._debug.body.push({
      kind: "line", bbox: box, text: r.recognized ? r.text : "",
      recognized: !!r.recognized, elements: [], cropUrl: r.cropUrl,
    });
  }
  capture.detected_elements = P.parseLines(lines);
  for (const d of capture._debug.body) {
    d.elements = capture.detected_elements
      .filter((e) => e.bbox && e.bbox[0] === d.bbox[0] && e.bbox[1] === d.bbox[1])
      .map((e) => (e.kind === "bullet" ? `bullet:${e.data.state}` : e.kind));
  }

  // ---- literal / as-is image regions (spec §16) ----
  for (const lit of g.literal_assets || []) {
    const r = await region(lit.bbox, false);
    capture._debug.body.push({
      kind: "literal", bbox: lit.bbox, confidence: lit.confidence, cropUrl: r.cropUrl,
    });
  }

  capture.notes = buildNotes(capture);
  return capture;
}

// keep the stored crop image reasonably small; OCR still ran on full resolution
function capCropUrl(cnv) {
  const maxW = 760;
  if (cnv.width <= maxW) return cnv.toDataURL("image/png");
  const s = maxW / cnv.width;
  const t = document.createElement("canvas");
  t.width = maxW;
  t.height = Math.max(1, Math.round(cnv.height * s));
  t.getContext("2d").drawImage(cnv, 0, 0, t.width, t.height);
  return t.toDataURL("image/png");
}

function buildNotes(c) {
  const nf = (c.detected_fiducials || []).length;
  const notes = [c.fiducial_mode === "corner_stickers"
    ? `${nf} adhesive corner sticker(s); orientation via geometry`
    : `${nf} ArUco marker(s); orientation via marker ids`];
  if (c.page_size_estimate) {
    const p = c.page_size_estimate;
    notes.push(`page ~${Math.round(p.width_mm)}x${Math.round(p.height_mm)} mm`
      + (p.best_match ? ` (${p.best_match})` : " (no standard match)"));
  }
  if (c.sharpness) {
    const soft = (c.sharpness.probes || []).filter((p) => !p.sharp).map((p) => p.name);
    notes.push(
      `sharpness ${c.sharpness.score} (lapvar ${c.sharpness.laplacian_variance})`
      + (soft.length ? `; soft at ${soft.join(", ")}` : "")
      + (c.sharpness.blurry ? " - BLURRY, retake" : ""),
    );
  }
  if (c.metadata_block) {
    notes.push(`metadata block via ${c.metadata_block.detection}: ${c.metadata_block.row1_cells.length}+${c.metadata_block.row2_cells.length} cells (conf ${c.metadata_block.confidence})`);
  }
  if (c.literal_assets && c.literal_assets.length) {
    notes.push(`${c.literal_assets.length} literal image region(s) detected and masked (spec §16)`);
  }
  const els = c.detected_elements || [];
  if (els.length) {
    const byKind = {};
    for (const e of els) byKind[e.kind] = (byKind[e.kind] || 0) + 1;
    notes.push("parsed elements: " + Object.entries(byKind).sort().map(([k, v]) => `${v} ${k}`).join(", "));
  }
  notes.push(`text backend: ${c.text_backend}`);
  return notes;
}

const esc = (s) => String(s == null ? "" : s).replace(/[<>&]/g, (m) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[m]));

// the rectified page with green boxes over every OpenCV-detected region
function annotateRectified(capture) {
  const cnv = document.createElement("canvas");
  cnv.width = rectCanvas.width;
  cnv.height = rectCanvas.height;
  const ctx = cnv.getContext("2d");
  ctx.drawImage(rectCanvas, 0, 0);
  ctx.lineWidth = Math.max(2, Math.round(cnv.width / 500));
  const box = (b, color) => { if (!b) return; ctx.strokeStyle = color; ctx.strokeRect(b[0], b[1], b[2], b[3]); };

  const mb = capture.metadata_block;
  if (mb) {
    box(mb.bbox, "rgba(46,204,113,0.95)");
    for (const c of [...mb.row1_cells, ...mb.row2_cells]) box(c, "rgba(46,204,113,0.75)");
    // registration marks: [x, y, size, acutance] — amber if that probe is soft
    for (const [mx, my, sz, ac] of mb.registration_marks || []) {
      ctx.strokeStyle = ac >= 0.3 ? "rgba(46,204,113,0.95)" : "rgba(242,193,78,0.95)";
      ctx.strokeRect(mx - sz / 2, my - sz / 2, sz, sz);
    }
  }
  for (const lb of capture.line_boxes || []) box(lb, "rgba(46,204,113,0.9)");
  for (const lit of capture.literal_assets || []) box(lit.bbox, "rgba(90,160,255,0.95)");
  return cnv.toDataURL("image/jpeg", JPEG_QUALITY);
}

function showResult({ rawUrl, rectUrl, jsonUrl, capture, stamp, error }) {
  $("shot-raw").src = rawUrl;
  const figRect = $("fig-rect");
  const dlRect = $("dl-rect");
  if (rectUrl) {
    $("shot-rect").src = annotateRectified(capture);
    figRect.classList.remove("empty");
    dlRect.style.display = "";
  } else {
    figRect.classList.add("empty");
    dlRect.style.display = "none";
  }

  const slug = stamp.toISOString().replace(/[:.]/g, "-");
  wireDownload("dl-raw", rawUrl, `wjm-${slug}.jpg`);
  wireDownload("dl-json", jsonUrl, `wjm-${slug}.json`);
  if (rectUrl) wireDownload("dl-rect", rectUrl, `wjm-${slug}-rectified.jpg`);

  $("sec-meta").replaceChildren(...sectionExtraction(capture, error));
  $("sec-debug-meta").replaceChildren(...sectionMetadataCrops(capture));
  $("sec-debug-body").replaceChildren(...sectionBody(capture));

  const rpt = $("result");
  rpt.hidden = false;
  rpt.scrollTop = 0;
}

const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const srow = (label, value, cls) =>
  el("div", "srow" + (cls ? " " + cls : ""), `<span class="slabel">${esc(label)}</span><span class="sval">${value}</span>`);

// --- section: parsed extraction values -------------------------------
function sectionExtraction(c, error) {
  const out = [el("h3", null, "Extraction")];
  if (error) out.push(srow("error", esc(error), "err"));
  out.push(srow("markers", `${(c.detected_fiducials || []).map((f) => f.id).join(", ") || "—"}  (${esc(c.dictionary || "DICT_4X4_50")})`));
  if (c.fiducial_mode === "corner_stickers") out.push(srow("fiducials", "adhesive corner stickers"));
  if (c.source) out.push(srow("photo", `${c.source.width}×${c.source.height}`));
  if (c.normalized) out.push(srow("rectified", `${c.normalized.width}×${c.normalized.height}`));
  const ps = c.page_size_estimate;
  if (ps) {
    out.push(srow(
      "page size (est)",
      `${Math.round(ps.width_mm)}×${Math.round(ps.height_mm)} mm`
      + (ps.best_match ? `  ≈ ${esc(ps.best_match)}` : "  (no standard match)")
      + `  ±${ps.match_error_mm}`,
    ));
  }
  if (c.metadata_block) out.push(srow("block found via", esc(c.metadata_block.detection)));
  out.push(srow("OCR engine", esc(c.text_backend || ocrBackend)));

  const s = c.sharpness;
  if (s) {
    out.push(srow(
      "sharpness",
      `${s.score}  (lapvar ${s.laplacian_variance})` + (s.blurry ? "  ⚠ blurry" : ""),
      s.blurry ? "err" : "",
    ));
    if (s.rectified_score != null) out.push(srow("  rectified score", `${s.rectified_score}`, "muted"));
    for (const p of s.probes || []) {
      out.push(srow("  " + p.name, `${p.acutance}  ${p.sharp ? "sharp" : "blurry"}`, p.sharp ? "muted" : "err"));
    }
  }

  const md = c.page_metadata || {};
  out.push(el("div", "srow", `<span class="slabel">page metadata</span><span class="sval">conf ${md._confidence ?? 0}</span>`));
  for (const f of METADATA_FIELDS) {
    const v = f === "topic_tags"
      ? (md.topic_tags && md.topic_tags.length ? md.topic_tags.join(", ") : "—")
      : (md[f] || "—");
    out.push(srow("  " + f, esc(v), md[f] || (f === "topic_tags" && md.topic_tags && md.topic_tags.length) ? "" : "muted"));
  }

  const els = c.detected_elements || [];
  if (els.length) {
    const byKind = {};
    for (const e of els) byKind[e.kind] = (byKind[e.kind] || 0) + 1;
    out.push(srow("elements", Object.entries(byKind).sort().map(([k, v]) => `${v} ${k}`).join("  ")));
  } else {
    out.push(srow("elements", "none parsed", "muted"));
  }
  return out;
}

// --- section: the 7 metadata cell crops fed to OCR -------------------
function sectionMetadataCrops(c) {
  const out = [el("h3", null, "Metadata cells — crops fed to OCR")];
  const cells = (c._debug && c._debug.metadata_cells) || [];
  if (!cells.length || !cells.some((x) => x.bbox)) {
    out.push(srow("", "no metadata block detected on this page", "muted"));
    return out;
  }
  for (const cell of cells) out.push(cropCard({
    label: cell.field,
    badge: !cell.bbox ? "no cell" : cell.recognized ? `${Math.round(cell.confidence * 100)}%` : "nothing read",
    badgeCls: !cell.bbox || !cell.recognized ? "none" : "",
    cropUrl: cell.cropUrl,
    read: cell.bbox ? (cell.recognized ? cell.text : "(nothing read)") : "(cell not found)",
    readEmpty: !cell.recognized,
  }));
  return out;
}

// --- section: page body — nodes / text lines / image blocks ---------
function sectionBody(c) {
  const out = [el("h3", null, "Page body — text lines & image regions")];
  const items = (c._debug && c._debug.body) || [];
  if (!items.length) {
    out.push(srow("", "no body regions segmented", "muted"));
    return out;
  }
  for (const it of items) {
    if (it.kind === "literal") {
      out.push(cropCard({
        label: "image block", badge: "as-is image", badgeCls: "img",
        cropUrl: it.cropUrl, read: `literal region (spec §16) — conf ${it.confidence}`, readEmpty: false,
      }));
      continue;
    }
    const kinds = it.elements && it.elements.length ? it.elements.join(", ") : (it.recognized ? "text" : "unread");
    out.push(cropCard({
      label: "line", badge: kinds, badgeCls: it.recognized ? "" : "none",
      cropUrl: it.cropUrl,
      read: it.recognized ? it.text : "(nothing read)",
      readEmpty: !it.recognized,
    }));
  }
  return out;
}

function cropCard({ label, badge, badgeCls, cropUrl, read, readEmpty }) {
  const card = el("div", "crop");
  card.appendChild(el("header", null,
    `<span>${esc(label)}</span><span class="badge ${badgeCls || ""}">${esc(badge)}</span>`));
  if (cropUrl) {
    const img = el("img");
    img.src = cropUrl;
    img.alt = label + " crop";
    card.appendChild(img);
  }
  card.appendChild(el("div", "read" + (readEmpty ? " empty" : ""), esc(read)));
  return card;
}

const wireDownload = (id, url, name) => {
  const a = $(id);
  a.href = url;
  a.download = name;
};

$("btn-retake").addEventListener("click", () => {
  $("result").hidden = true;                 // reveals the full-screen camera view
  while (objectUrls.length) URL.revokeObjectURL(objectUrls.pop());
  cropUrls.length = 0;
  $("sec-debug-meta").replaceChildren();
  $("sec-debug-body").replaceChildren();
  running = true;
  lockStart = 0;
  markers = [];
  cooldownUntil = 0;
  coachMsg = "";
  softRoles = new Set();
  setStatus("Point at a WJM sheet");
  requestAnimationFrame(loop);
});

$("btn-shutter").addEventListener("click", () => {
  if (!busy && running) capture("manual");
});

$("btn-flip").addEventListener("click", () => {
  facing = facing === "environment" ? "user" : "environment";
  start();
});

// ---- misc --------------------------------------------------
function canvasUrl(canvas, type, quality) {
  return new Promise((resolve) => {
    canvas.toBlob((b) => resolve(URL.createObjectURL(b)), type, quality);
  });
}
const round1 = (v) => Math.round(v * 10) / 10;
const once = (el, ev) => new Promise((res) => el.addEventListener(ev, res, { once: true }));

function fatal(msg) {
  running = false;
  $("fatal-msg").textContent = msg;
  $("fatal").hidden = false;
  $("gate").hidden = true;
}

window.addEventListener("beforeunload", stopStream);
