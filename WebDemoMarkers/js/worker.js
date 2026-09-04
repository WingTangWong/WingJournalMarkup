/* WebDemoMarkers vision worker (classic worker) — OpenCV.js runs here so the
 * camera preview never stutters. See js/glyph-detect.js for the actual
 * detection algorithm; this file is just the load/protocol plumbing, mirrored
 * off MobileDeviceDemo/js/worker.js.
 *
 * Protocol
 *   main → { type:"init", opencvUrls:[...] }
 *   ←    { type:"ready" } | { type:"error", message }
 *   main → { type:"detect", seq, width, height, buffer }   // live frame (RGBA)
 *   ←    { type:"glyphs", seq, glyphs }
 */
"use strict";

let cv = null;
let G = null; // WJMGlyphs

self.onmessage = (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") return void init(msg);
    if (msg.type === "detect") return void onDetect(msg);
  } catch (err) {
    post({ type: "error", message: String((err && err.message) || err) });
  }
};

const post = (obj, transfer) => self.postMessage(obj, transfer || []);
const here = (rel) => new URL(rel, self.location.href).href;

async function init(msg) {
  let loaded = null;
  for (const url of msg.opencvUrls) {
    try {
      self.cv = {};
      importScripts(url);
      await waitForRuntime();
      loaded = url;
      break;
    } catch (err) { /* try next */ }
  }
  if (!loaded) return post({ type: "error", message: "could not load any OpenCV.js build" });
  cv = self.cv;

  importScripts(here("glyph-detect.js"));
  G = self.WJMGlyphs;

  post({ type: "ready", build: loaded });
}

function waitForRuntime() {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 120_000;
    const tick = () => {
      const m = self.cv;
      if (m && m.Mat && typeof m.matFromImageData === "function") return resolve();
      if (Date.now() > deadline) return reject(new Error("OpenCV.js init timed out"));
      setTimeout(tick, 50);
    };
    try { if (self.cv && typeof self.cv === "object") self.cv.onRuntimeInitialized = tick; }
    catch (e) { /* */ }
    tick();
  });
}

function onDetect({ seq, width, height, buffer }) {
  if (!cv || !G) return post({ type: "glyphs", seq, glyphs: [] });
  const src = cv.matFromImageData({ data: new Uint8ClampedArray(buffer), width, height });
  const gray = new cv.Mat();
  let glyphs = [];
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    glyphs = G.detectGlyphs(cv, gray);
  } finally {
    src.delete();
    gray.delete();
  }
  post({ type: "glyphs", seq, glyphs });
}
