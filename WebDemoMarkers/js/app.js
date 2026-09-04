// WJM WebDemoMarkers — a stand-alone test bench for the bullseye + 3-bit
// corner-code glyph (js/glyph-detect.js), independent of the ArUco-based
// MobileDeviceDemo. Live camera preview here on the main thread; OpenCV.js +
// the actual detection run in js/worker.js so the preview never stutters.
"use strict";

const OPENCV_URLS = (window.OPENCV_URLS || ["vendor/opencv.js"]).map(
  (u) => new URL(u, location.href).href,
);

const PROC_W = 900;   // detection runs on a frame this wide
const TICK_MS = 80;   // min gap between detection requests

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");

const procCanvas = document.createElement("canvas");
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

let worker = null;
let stream = null;
let track = null;
let facing = "environment";
let torchOn = false;

let running = false;
let pending = false;
let seq = 0;
let lastSent = 0;
let procScale = 1;
let latestGlyphs = [];

function fatal(msg) {
  $("fatal-msg").textContent = msg;
  $("fatal").hidden = false;
  $("gate").hidden = true;
}

// ---- boot: load OpenCV.js in the worker, then unlock the start button ----
function boot() {
  worker = new Worker("js/worker.js");
  worker.onmessage = (e) => onWorkerMessage(e.data);
  worker.onerror = (e) => fatal(`Worker error: ${e.message || e}`);
  worker.postMessage({ type: "init", opencvUrls: OPENCV_URLS });
}

function onWorkerMessage(msg) {
  if (msg.type === "ready") {
    $("status").textContent = "Ready — tap Start";
    const btn = $("btn-start");
    btn.disabled = false;
    btn.textContent = "Start camera";
    return;
  }
  if (msg.type === "error") {
    fatal(msg.message);
    return;
  }
  if (msg.type === "glyphs") {
    pending = false;
    latestGlyphs = msg.glyphs || [];
  }
}

// ---- camera -------------------------------------------------------------
async function start() {
  try {
    $("btn-start").disabled = true;
    stopStream();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facing },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    });
    track = stream.getVideoTracks()[0];
    video.srcObject = stream;
    await video.play().catch(() => {});
    if (video.readyState < 1) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
    await tryContinuousFocus();

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    procCanvas.width = Math.min(PROC_W, vw);
    procCanvas.height = Math.round(procCanvas.width * (vh / vw));
    procScale = procCanvas.width / vw;

    setupTorchButton();
    $("gate").hidden = true;
    $("controls").hidden = false;
    running = true;
    pending = false;
    latestGlyphs = [];
    requestAnimationFrame(loop);
  } catch (err) {
    $("btn-start").disabled = false;
    if (err && err.name === "NotAllowedError") {
      $("gate-fine").textContent = "Camera permission was denied. Enable it for this site and reload.";
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

async function tryContinuousFocus() {
  if (!track || !track.getCapabilities) return;
  let caps;
  try { caps = track.getCapabilities(); } catch { return; }
  const want = {};
  if (Array.isArray(caps.focusMode) && caps.focusMode.includes("continuous")) want.focusMode = "continuous";
  if (Array.isArray(caps.exposureMode) && caps.exposureMode.includes("continuous")) want.exposureMode = "continuous";
  if (!Object.keys(want).length) return;
  try { await track.applyConstraints(want); } catch { /* keep default */ }
}

function setupTorchButton() {
  const btn = $("btn-torch");
  const caps = track && track.getCapabilities ? track.getCapabilities() : {};
  if (caps && caps.torch) {
    btn.hidden = false;
    btn.onclick = async () => {
      torchOn = !torchOn;
      btn.setAttribute("aria-pressed", String(torchOn));
      try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch { /* ignore */ }
    };
  } else {
    btn.hidden = true;
  }
}

$("btn-flip").onclick = () => {
  facing = facing === "environment" ? "user" : "environment";
  if (running) start();
};

// ---- main loop ------------------------------------------------------------
function loop(ts) {
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;

  if (!pending && ts - lastSent >= TICK_MS) {
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
  drawOverlay(latestGlyphs, view);
  updateStatus(latestGlyphs);
}

function updateStatus(glyphs) {
  const el = $("status");
  if (!glyphs.length) {
    el.textContent = "Looking for glyphs…";
    el.className = "status";
    return;
  }
  const values = glyphs.map((g) => g.value).join(", ");
  el.textContent = `${glyphs.length} glyph${glyphs.length > 1 ? "s" : ""} — value ${values}`;
  el.className = "status ok";
}

// ---- video <-> screen mapping (video is object-fit:cover) ---------------
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
const intrinsicToScreen = ([x, y], v) => [(x - v.visX) * v.s, (y - v.visY) * v.s];

// ---- overlay --------------------------------------------------------------
function drawOverlay(glyphs, view) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(view.cssW * dpr);
  const H = Math.round(view.cssH * dpr);
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, view.cssW, view.cssH);

  const toScreen = (x, y) => intrinsicToScreen([x / procScale, y / procScale], view);

  for (const g of glyphs) {
    const pts = g.quad.map(([x, y]) => toScreen(x, y));

    octx.strokeStyle = "#2ecc71";
    octx.lineWidth = 3;
    octx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? octx.moveTo(x, y) : octx.lineTo(x, y)));
    octx.closePath();
    octx.stroke();

    // bullseye
    const [bx, by] = toScreen(g.bullseye.cx, g.bullseye.cy);
    const br = Math.max(4, (g.bullseye.diameter / procScale) * view.s * 0.5);
    octx.beginPath();
    octx.arc(bx, by, br, 0, Math.PI * 2);
    octx.stroke();

    // the anchor corner itself (nearest the bullseye) — small ring, not a bit
    octx.beginPath();
    octx.arc(pts[g.anchorIndex][0], pts[g.anchorIndex][1], 5, 0, Math.PI * 2);
    octx.strokeStyle = "#2ecc71";
    octx.lineWidth = 1.5;
    octx.stroke();

    // the 3 code corners, in UR/LR/LL order — filled = 1, hollow = 0
    const codePts = [pts[(g.anchorIndex + 1) % 4], pts[(g.anchorIndex + 2) % 4], pts[(g.anchorIndex + 3) % 4]];
    const bitVals = [g.bits.upperRight, g.bits.lowerRight, g.bits.lowerLeft];
    codePts.forEach(([x, y], i) => {
      octx.beginPath();
      octx.arc(x, y, 5, 0, Math.PI * 2);
      octx.fillStyle = bitVals[i] ? "#2ecc71" : "rgba(46,204,113,0.15)";
      octx.fill();
      octx.lineWidth = 1.5;
      octx.strokeStyle = "#2ecc71";
      octx.stroke();
    });

    // decoded value, centered on the box
    const cx = pts.reduce((s, p) => s + p[0], 0) / 4;
    const cy = pts.reduce((s, p) => s + p[1], 0) / 4;
    const label = String(g.value);
    octx.font = "700 24px -apple-system, BlinkMacSystemFont, sans-serif";
    octx.textAlign = "center";
    octx.textBaseline = "middle";
    octx.lineWidth = 4;
    octx.strokeStyle = "rgba(0,0,0,0.85)";
    octx.strokeText(label, cx, cy);
    octx.fillStyle = "#2ecc71";
    octx.fillText(label, cx, cy);
  }
}

// ---- wire up --------------------------------------------------------------
$("btn-start").onclick = start;
boot();
