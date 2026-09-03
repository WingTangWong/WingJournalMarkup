/* WJM detection worker (classic worker).
 *
 * Keeps the 11 MB OpenCV.js WASM compile and every per-frame detection off the
 * main thread, so the camera preview and overlay never stutter.
 *
 * Protocol
 *   main → { type:"init", opencvUrls:[...] }
 *   ←    { type:"ready", hasAruco } | { type:"error", message }
 *   main → { type:"detect",  seq, width, height, buffer }        // RGBA, buffer transferred
 *   ←    { type:"markers",  seq, markers }
 *   main → { type:"rectify", seq, width, height, buffer, markers }  // RGBA
 *   ←    { type:"rectified", seq, width, height, buffer, quad } | { type:"rectified", seq, quad:null }
 */
"use strict";

let cv = null;
let V = null;
let detector = null;

self.onmessage = (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") return void init(msg.opencvUrls);
    if (msg.type === "detect") return void onDetect(msg);
    if (msg.type === "rectify") return void onRectify(msg);
  } catch (err) {
    post({ type: "error", message: String(err && err.message || err) });
  }
};

function post(obj, transfer) {
  self.postMessage(obj, transfer || []);
}

async function init(opencvUrls) {
  let loaded = null;
  for (const url of opencvUrls) {
    try {
      self.cv = {}; // classic-build Module seed
      importScripts(url);
      await waitForRuntime();
      loaded = url;
      break;
    } catch (err) {
      // try the next candidate
    }
  }
  if (!loaded) {
    return post({ type: "error", message: "could not load any OpenCV.js build" });
  }
  cv = self.cv;

  importScripts(new URL("vision-core.js", self.location.href).href);
  V = self.WJMVision;

  if (!V.hasAruco(cv)) {
    return post({
      type: "error",
      message:
        "This OpenCV.js build has no ArUco bindings. Vendor an objdetect-enabled " +
        "build to vendor/opencv.js (see README / fetch-opencv.sh).",
    });
  }
  detector = new V.MarkerDetector(cv);
  post({ type: "ready", hasAruco: true, build: loaded });
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

function matFromBuffer(buffer, width, height) {
  return cv.matFromImageData({
    data: new Uint8ClampedArray(buffer),
    width,
    height,
  });
}

function onDetect({ seq, width, height, buffer }) {
  if (!detector) return post({ type: "markers", seq, markers: [] });
  const src = matFromBuffer(buffer, width, height);
  const gray = new cv.Mat();
  let markers = [];
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    markers = detector.detect(gray);
  } finally {
    src.delete();
    gray.delete();
  }
  post({ type: "markers", seq, markers });
}

function onRectify({ seq, width, height, buffer, markers }) {
  const quad = V.pageQuadFromMarkers(markers || []);
  if (!quad) return post({ type: "rectified", seq, quad: null });

  const src = matFromBuffer(buffer, width, height);
  try {
    const { mat, width: rw, height: rh } = V.rectify(cv, src, quad);
    const out = new Uint8ClampedArray(mat.data); // copy out of the WASM heap
    mat.delete();
    post(
      { type: "rectified", seq, width: rw, height: rh, buffer: out.buffer, quad },
      [out.buffer],
    );
  } catch (err) {
    post({ type: "rectified", seq, quad: null, error: String(err && err.message || err) });
  } finally {
    src.delete();
  }
}
