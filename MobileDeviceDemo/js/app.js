// WJM MobileDeviceDemo — auto-capture a page the moment its four corner markers
// sit inside the on-screen guide.
//
// The camera preview, overlay and UI run on the main thread; OpenCV.js (WASM)
// and every ArUco detection run in js/worker.js, so nothing here ever blocks.

const OPENCV_URLS = (window.OPENCV_URLS || ["vendor/opencv.js"]).map(
  (u) => new URL(u, location.href).href,
);
const ROLE_ORDER = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"];

// ---- tunables ----------------------------------------------------------
const PROC_W = 900;           // detection runs on a frame this wide
const TICK_MS = 70;           // min gap between detection requests
const HOLD_MS = 450;          // all-4-inside must persist this long to fire
const STALE_MS = 260;         // ignore marker results older than this for the trigger
const EDGE_SLACK = 0.012;     // fraction of guide width tolerated outside the box
const MIN_MARKER_FRAC = 0.018; // reject markers smaller than this * guide width
const JPEG_QUALITY = 0.92;

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
const rectifyWaiters = new Map();

const procCanvas = document.createElement("canvas");
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });
const grabCanvas = document.createElement("canvas");
const objectUrls = [];

// ---- boot ---------------------------------------------------------
(function boot() {
  setStatus("Loading the vision engine…");
  worker = new Worker("js/worker.js");
  worker.onerror = (e) => fatal(`worker failed: ${e.message || e}`);
  worker.onmessage = onWorkerMessage;
  worker.postMessage({ type: "init", opencvUrls: OPENCV_URLS });
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
      break;
    }
    case "rectified": {
      const w = rectifyWaiters.get(m.seq);
      if (w) { rectifyWaiters.delete(m.seq); w(m); }
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
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    });
    track = stream.getVideoTracks()[0];
    video.srcObject = stream;
    await video.play().catch(() => {});
    // a canvas/stream source can reach metadata before we could attach a listener
    if (video.readyState < 1) await once(video, "loadedmetadata");

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

  updatePips(goodRoles);
  drawOverlay(assessed, guide, view, allIn && fresh);
  updateStatus(assessed, goodRoles, allIn && fresh);

  if (allIn && fresh && $("chk-auto").checked && !busy) {
    if (!lockStart) lockStart = ts;
    const held = ts - lockStart;
    setLockRing(Math.min(1, held / HOLD_MS));
    if (held >= HOLD_MS) capture("auto");
  } else {
    lockStart = 0;
    setLockRing(0);
  }
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
function drawOverlay(assessed, guide, view, locked) {
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

  // corner brackets
  const col = locked ? "#2ecc71" : "#f2c14e";
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
    const ok = a.inside && a.bigEnough && a.marker.role !== null;
    octx.strokeStyle = ok ? "#2ecc71" : "#e5533d";
    octx.fillStyle = ok ? "rgba(46,204,113,0.22)" : "rgba(229,83,61,0.16)";
    octx.lineWidth = 3;
    octx.fill();
    octx.stroke();

    const [lx, ly] = intrinsicToScreen(a.marker.center, view);
    octx.fillStyle = "#fff";
    octx.font = "700 12px -apple-system, system-ui, sans-serif";
    octx.textAlign = "center";
    octx.fillText(a.marker.role ? a.marker.role.replace("_", " ") : `#${a.marker.id}`, lx, ly + 4);
  }
}

function updatePips(goodRoles) {
  for (const [role, el] of Object.entries(pips)) el.classList.toggle("on", goodRoles.has(role));
}

function updateStatus(assessed, goodRoles) {
  const n = assessed.length;
  if (goodRoles.size === 4) {
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

  // ask the worker to perspective-normalize (it computes the page quad itself)
  const full = gctx.getImageData(0, 0, vw, vh);
  const rectSeq = ++seq;
  const rectMsg = await new Promise((resolve) => {
    rectifyWaiters.set(rectSeq, resolve);
    worker.postMessage(
      {
        type: "rectify", seq: rectSeq,
        width: vw, height: vh, buffer: full.data.buffer,
        markers: capturedMarkers,
      },
      [full.data.buffer],
    );
    setTimeout(() => {
      if (rectifyWaiters.has(rectSeq)) { rectifyWaiters.delete(rectSeq); resolve({ error: "timeout" }); }
    }, 8000);
  });

  let rectUrl = null;
  let normalized = null;
  if (rectMsg && rectMsg.buffer) {
    const rc = document.createElement("canvas");
    rc.width = rectMsg.width;
    rc.height = rectMsg.height;
    rc.getContext("2d").putImageData(
      new ImageData(new Uint8ClampedArray(rectMsg.buffer), rectMsg.width, rectMsg.height),
      0, 0,
    );
    rectUrl = await canvasUrl(rc, "image/png");
    normalized = { width: rectMsg.width, height: rectMsg.height, target_long_px: 1600 };
  }

  const rawUrl = await canvasUrl(grabCanvas, "image/jpeg", JPEG_QUALITY);
  const stamp = new Date();
  const sidecar = {
    app: "WJM MobileDeviceDemo",
    captured_at: stamp.toISOString(),
    trigger,
    engine: "OpenCV.js (WebAssembly, Web Worker)",
    dictionary: "DICT_4X4_50",
    image: { width: vw, height: vh },
    markers: capturedMarkers.map((m) => ({
      id: m.id,
      role: m.role,
      corners: m.corners.map(([x, y]) => [round1(x), round1(y)]),
    })),
    page_frame_quad: rectMsg && rectMsg.quad
      ? rectMsg.quad.map(([x, y]) => [round1(x), round1(y)])
      : null,
    normalized,
  };
  const jsonUrl = URL.createObjectURL(
    new Blob([JSON.stringify(sidecar, null, 2)], { type: "application/json" }),
  );

  objectUrls.push(rawUrl, jsonUrl);
  if (rectUrl) objectUrls.push(rectUrl);
  showResult({ rawUrl, rectUrl, jsonUrl, sidecar, stamp });
  busy = false;
}

function showResult({ rawUrl, rectUrl, jsonUrl, sidecar, stamp }) {
  $("shot-raw").src = rawUrl;
  const figRect = $("fig-rect");
  const dlRect = $("dl-rect");
  if (rectUrl) {
    $("shot-rect").src = rectUrl;
    figRect.classList.remove("empty");
    dlRect.style.display = "";
  } else {
    figRect.classList.add("empty");
    dlRect.style.display = "none";
  }

  const slug = stamp.toISOString().replace(/[:.]/g, "-");
  wireDownload("dl-raw", rawUrl, `wjm-${slug}.jpg`);
  wireDownload("dl-json", jsonUrl, `wjm-${slug}.json`);
  if (rectUrl) wireDownload("dl-rect", rectUrl, `wjm-${slug}-rectified.png`);

  const m = sidecar;
  $("result-meta").textContent =
    `trigger    ${m.trigger}\n` +
    `markers    ${m.markers.map((x) => x.id).join(", ") || "—"}  (${m.dictionary})\n` +
    `photo      ${m.image.width}×${m.image.height}\n` +
    `rectified  ${m.normalized ? `${m.normalized.width}×${m.normalized.height}` : "—"}\n` +
    `engine     ${m.engine}`;

  $("result").hidden = false;
}

const wireDownload = (id, url, name) => {
  const a = $(id);
  a.href = url;
  a.download = name;
};

$("btn-retake").addEventListener("click", () => {
  $("result").hidden = true;
  while (objectUrls.length) URL.revokeObjectURL(objectUrls.pop());
  running = true;
  lockStart = 0;
  markers = [];
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
