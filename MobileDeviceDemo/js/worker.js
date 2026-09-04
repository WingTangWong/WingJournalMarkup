/* WJM vision worker (classic worker).
 *
 * OpenCV.js (WASM) + ArUco + all the *geometric* extraction run here, so the
 * camera preview and overlay on the main thread never stutter. Text recognition
 * (Tesseract.js) runs on the main thread — it manages its own worker and needs a
 * canvas it can't get here.
 *
 * Protocol (SCANNER.md)
 *   main → { type:"init", opencvUrls:[...] }
 *   ←    { type:"ready", hasAruco } | { type:"error", message }
 *   main → { type:"detect", seq, width, height, buffer }          // live frame
 *   ←    { type:"markers", seq, markers }
 *   main → { type:"analyze", seq, frames:[{buffer,width,height}], markers, mode }  // BASE burst
 *   ←    { type:"analyzed", seq, geometry, rectified, targets }   // geometry.targets = close-up plan
 *   main → { type:"composite", seq, targetIndex, frames, markers } // CLOSE-UP burst
 *   ←    { type:"composited", seq, ok, method, detail, rectified }
 *   main → { type:"finish", seq }                                  // recognise the final canvas
 *   ←    { type:"analyzed", seq, geometry, rectified }
 *   ←    { type:"progress", seq, stage }  /  { type:"analyzed"|"composited", seq, error }
 *
 * Phase A: a burst is scored frame-by-frame (Tenengrad); the sharpest is kept
 * and rectified into a fixed 8.5x11 @ 300 DPI canvas. No accept/reject gate.
 * Phase B: `analyze` also keeps that canvas in a session and plans up to five
 * close-up targets; each `composite` re-shoots one and warps it in (ArUco
 * homography, ORB fallback); `finish` recognises the composited canvas.
 */
"use strict";

let cv = null;
let V = null;   // WJMVision
let detector = null;

// the fixed rectified canvas: 8.5 x 11 in at 300 DPI (SCANNER.md)
const LETTER_PX = [2550, 3300];

// live compositing session: the base canvas + its fiducials in canvas space
let session = { canvas: null, baseCorners: {}, targets: [], landmarks: [], stickerMode: false };

function resetSession() {
  if (session.canvas) { try { session.canvas.delete(); } catch (e) { /* */ } }
  session = { canvas: null, baseCorners: {}, targets: [], landmarks: [], stickerMode: false };
}

self.onmessage = (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") return void init(msg);
    if (msg.type === "detect") return void onDetect(msg);
    if (msg.type === "analyze") return void onAnalyze(msg);
    if (msg.type === "composite") return void onComposite(msg);
    if (msg.type === "finish") return void onFinish(msg);
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

  importScripts(here("vision-core.js"));
  V = self.WJMVision;

  if (!V.hasAruco(cv)) {
    return post({
      type: "error",
      message: "This OpenCV.js build has no ArUco bindings (see README / fetch-opencv.sh).",
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

const matFromBuffer = (buffer, width, height) =>
  cv.matFromImageData({ data: new Uint8ClampedArray(buffer), width, height });

const grayOf = (rgba) => {
  const g = new cv.Mat();
  cv.cvtColor(rgba, g, cv.COLOR_RGBA2GRAY);
  return g;
};
const round3 = (v) => Math.round(v * 1000) / 1000;

function onDetect({ seq, width, height, buffer }) {
  if (!detector) return post({ type: "markers", seq, markers: [] });
  const src = matFromBuffer(buffer, width, height);
  const gray = grayOf(src);
  let markers = [];
  let sharpness = null;
  let mode = "sheet";
  try {
    const all = detector.detect(gray);
    const sheet = all.filter((m) => m.id >= 0 && m.id <= 3);
    const stkm = all.filter((m) => m.id === V.CORNER_STICKER_ID);
    if (stkm.length >= 3 && sheet.length < 3) {
      // adhesive-sticker page: geometry gives the roles (spec §11.2)
      mode = "stickers";
      markers = V.detectCornerStickers(cv, gray, stkm).map((s) => ({
        id: V.CORNER_STICKER_ID, role: s.role,
        corners: s.marker.corners, center: s.marker.center,
      }));
    } else {
      markers = sheet.length ? sheet : all;
    }
    sharpness = V.assessSharpness(cv, gray, markers, []);
  } finally { src.delete(); gray.delete(); }
  post({ type: "markers", seq, markers, sharpness, mode });
}

// pick the sharpest frame of a burst (Tenengrad); returns kept Mats + meta,
// deletes the losers. Caller deletes best.src / best.gray.
function pickSharpest(frames) {
  const scores = [];
  let best = null;
  frames.forEach((f, i) => {
    const src = matFromBuffer(f.buffer, f.width, f.height);
    const gray = grayOf(src);
    const score = V.tenengrad(cv, gray);
    scores.push(Math.round(score * 100) / 100);
    if (!best || score > best.score) {
      if (best) { best.src.delete(); best.gray.delete(); }
      best = { src, gray, score, index: i, width: f.width, height: f.height };
    } else {
      src.delete(); gray.delete();
    }
  });
  best.scores = scores;
  return best;
}

// the recognition-geometry tail, run on a normalized canvas (base or composited)
function recognizeGeometry(normalized, grayN, keep, progress) {
  const normMarkers = detector.detect(grayN);
  const markerBoxes = normMarkers.map((m) => {
    const xs = m.corners.map((p) => p[0]);
    const ys = m.corners.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys),
      Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
  });

  progress("literal regions");
  const literals = V.detectLiteralAssets(cv, grayN);
  const forParsing = keep(normalized.clone());
  if (literals.length) V.maskLiterals(cv, forParsing, literals);
  const grayMasked = keep(grayOf(forParsing));

  progress("metadata block");
  const block = V.detectMetadataBlock(cv, grayMasked, 0.42, markerBoxes, normMarkers);

  progress("sharpness");
  const regMarks = block && block.registration_marks
    ? block.registration_marks.map((m) => ({ center: [m[0], m[1]], size: m[2], acutance: m[3] }))
    : [];
  const sharpness = V.assessSharpness(cv, grayN, normMarkers, regMarks);

  progress("segmenting text");
  const nh = grayN.rows;
  const skipTop = block ? (block.bbox[1] + block.bbox[3]) / nh : 0;
  const lineBoxes = V.segmentLines(cv, grayMasked).filter(([, y]) => y >= skipTop * nh);

  return { normMarkers, block, literals, lineBoxes, sharpness };
}

function sendCanvas(type, seq, extra) {
  const buf = new Uint8ClampedArray(session.canvas.data);
  post(
    { type, seq, ...extra,
      rectified: { buffer: buf.buffer, width: session.canvas.cols, height: session.canvas.rows } },
    [buf.buffer],
  );
}

// ---- BASE capture --------------------------------------------------
function onAnalyze({ seq, frames, markers, mode }) {
  const progress = (stage) => post({ type: "progress", seq, stage });
  const trash = [];
  const keep = (m) => { trash.push(m); return m; };
  resetSession();
  try {
    if (!frames || !frames.length) {
      return post({ type: "analyzed", seq, error: "no frames in the burst" });
    }

    progress("choosing the sharpest frame");
    const best = pickSharpest(frames);
    const src = keep(best.src);
    const graySrc = keep(best.gray);

    // page frame from the winning frame's own fiducials
    let stickerMode = mode === "stickers";
    let quad;
    const all = detector.detect(graySrc);
    const sheet = all.filter((m) => m.id >= 0 && m.id <= 3);
    if (stickerMode || (sheet.length < 3 && all.some((m) => m.id === V.CORNER_STICKER_ID))) {
      stickerMode = true;
      const stkm = all.filter((m) => m.id === V.CORNER_STICKER_ID);
      quad = V.stickerQuad(V.detectCornerStickers(cv, graySrc, stkm));
    } else {
      quad = V.pageQuadFromMarkers(sheet.length >= 3 ? sheet : (markers || []));
    }
    if (!quad) return post({ type: "analyzed", seq, error: "need all four corner fiducials to normalize the page" });

    progress("rectify");
    const { mat: normalized, width: nw, height: nh } =
      V.rectify(cv, src, quad, null, LETTER_PX);
    keep(normalized);
    const grayN = keep(grayOf(normalized));

    let pageSize = null;
    if (stickerMode) {
      const nstk = detector.detect(grayN).filter((m) => m.id === V.CORNER_STICKER_ID);
      pageSize = V.estimatePageSize(V.detectCornerStickers(cv, grayN, nstk));
    }

    const geo = recognizeGeometry(normalized, grayN, keep, progress);
    geo.sharpness.focus_score = Math.round(best.score * 100) / 100;

    // keep the canvas + fiducials (canvas space) for the close-up passes
    session.canvas = normalized.clone();
    session.stickerMode = stickerMode;
    session.baseCorners = {};
    if (!stickerMode) {
      for (const m of geo.normMarkers) session.baseCorners[m.id] = m.corners;
    }
    // known text/content rects (canvas space) — used to steer feature
    // matching onto real landmarks instead of blank paper when a close-up
    // has too few (or no) shared ArUco anchors to register by directly
    session.landmarks = [
      ...(geo.block ? [...geo.block.row1_cells, ...geo.block.row2_cells] : []),
      ...geo.lineBoxes,
    ];
    progress("planning close-ups");
    session.targets = V.planTargets(nw, nh, geo.block, geo.literals, geo.lineBoxes);

    const shotFiducials = (stickerMode
      ? all.filter((m) => m.id === V.CORNER_STICKER_ID) : sheet
    ).map((m) => ({
      id: m.id, role: m.role || null,
      corners: m.corners.map(([x, y]) => [round3(x), round3(y)]),
      center: (m.center || [0, 0]).map(round3),
    }));

    sendCanvas("analyzed", seq, {
      targets: session.targets,
      geometry: {
        dictionary: "DICT_4X4_50",
        fiducial_mode: stickerMode ? "corner_stickers" : "printed_sheet",
        source: { width: best.width, height: best.height },
        burst: { count: frames.length, scores: best.scores, winner: best.index, focus: geo.sharpness.focus_score },
        page_frame_quad: quad.map(([x, y]) => [round3(x), round3(y)]),
        orientation: { method: stickerMode ? "geometry" : "aruco_ids", degrees: 0 },
        normalized: { width: nw, height: nh },
        page_size_estimate: pageSize,
        detected_fiducials: shotFiducials,
        base_markers_canvas: geo.normMarkers.map((m) => ({
          id: m.id,
          center: (m.center || [0, 0]).map(round3),
          corners: m.corners.map(([x, y]) => [round3(x), round3(y)]),
        })),
        metadata_block: geo.block,
        literal_assets: geo.literals,
        line_boxes: geo.lineBoxes,
        sharpness: geo.sharpness,
        targets: session.targets,
      },
    });
  } catch (err) {
    post({ type: "analyzed", seq, error: String((err && err.stack) || err) });
  } finally {
    for (const m of trash) { try { m.delete(); } catch (e) { /* */ } }
  }
}

// ---- CLOSE-UP capture: register + composite one target ------------
function onComposite({ seq, targetIndex, frames }) {
  const progress = (stage) => post({ type: "progress", seq, stage });
  const trash = [];
  const keep = (m) => { trash.push(m); return m; };
  try {
    if (!session.canvas) return post({ type: "composited", seq, error: "no base capture in session" });
    const rect = session.targets[targetIndex];
    if (!rect) return post({ type: "composited", seq, error: "unknown target " + targetIndex });
    if (!frames || !frames.length) return post({ type: "composited", seq, error: "no frames" });

    progress("choosing the sharpest close-up");
    const best = pickSharpest(frames);
    const src = keep(best.src);
    const graySrc = keep(best.gray);

    // registration cascade — each stage is rotation-tolerant on its own, so a
    // close-up taken with the phone held any way (including sideways for a
    // wide target) still lines up:
    //  1. shared ArUco anchors (exact, when the page has them and >=2 are in
    //     frame) — a homography from point correspondences carries rotation
    //     without any special-casing;
    //  2. AKAZE, restricted to known text/content landmarks on the base side
    //     (metadata cells + body lines) so matches land on real ink, not
    //     paper texture — handles blur well;
    //  3. ORB over the same landmarks, as a second try with more (cheaper,
    //     noisier) keypoints when AKAZE's stricter response found too few.
    // ORB/AKAZE keypoints each carry their own dominant orientation and their
    // descriptors are sampled relative to it, so no separate "try every
    // rotation" step is needed — that is what makes them rotation-tolerant.
    progress("registering the close-up");
    const closeMarkers = detector.detect(graySrc);
    let res = { H: null, method: "none" };
    if (!session.stickerMode) res = V.anchorHomography(cv, closeMarkers, session.baseCorners);
    if (!res.H) {
      const cg = keep(grayOf(session.canvas));
      progress("matching landmarks (AKAZE)");
      res = V.akazeHomography(cv, graySrc, cg, rect, session.landmarks);
      if (!res.H) {
        progress("matching landmarks (ORB)");
        res = V.orbHomography(cv, graySrc, cg, rect, session.landmarks);
      }
    }
    if (!res.H) {
      return post({
        type: "composited", seq, ok: false, method: res.method,
        detail: res.method === "anchors"
          ? `only ${res.shared || 0} shared marker(s) — keep two in view`
          : "couldn't line the close-up up with the page — try holding it steadier",
      });
    }

    progress("compositing");
    V.compositeInto(cv, session.canvas, src, res.H, rect, 14);
    res.H.delete();

    sendCanvas("composited", seq, {
      ok: true, method: res.method,
      detail: res.method === "anchors"
        ? `${res.shared} anchors` : `${res.method}, ${res.inliers} inliers`,
      focus: Math.round(best.score * 100) / 100,
    });
  } catch (err) {
    post({ type: "composited", seq, error: String((err && err.stack) || err) });
  } finally {
    for (const m of trash) { try { m.delete(); } catch (e) { /* */ } }
  }
}

// ---- recognise the final (composited) canvas ---------------------
function onFinish({ seq }) {
  const progress = (stage) => post({ type: "progress", seq, stage });
  const trash = [];
  const keep = (m) => { trash.push(m); return m; };
  try {
    if (!session.canvas) return post({ type: "analyzed", seq, error: "no session canvas" });
    const grayN = keep(grayOf(session.canvas));
    const geo = recognizeGeometry(session.canvas, grayN, keep, progress);
    const nstk = session.stickerMode
      ? detector.detect(grayN).filter((m) => m.id === V.CORNER_STICKER_ID) : [];
    const pageSize = session.stickerMode
      ? V.estimatePageSize(V.detectCornerStickers(cv, grayN, nstk)) : null;

    sendCanvas("analyzed", seq, {
      geometry: {
        dictionary: "DICT_4X4_50",
        fiducial_mode: session.stickerMode ? "corner_stickers" : "printed_sheet",
        orientation: { method: session.stickerMode ? "geometry" : "aruco_ids", degrees: 0 },
        normalized: { width: session.canvas.cols, height: session.canvas.rows },
        page_size_estimate: pageSize,
        detected_fiducials: geo.normMarkers.map((m) => ({
          id: m.id, role: m.role || null,
          corners: m.corners.map(([x, y]) => [round3(x), round3(y)]),
          center: (m.center || [0, 0]).map(round3),
        })),
        metadata_block: geo.block,
        literal_assets: geo.literals,
        line_boxes: geo.lineBoxes,
        sharpness: geo.sharpness,
      },
    });
  } catch (err) {
    post({ type: "analyzed", seq, error: String((err && err.stack) || err) });
  } finally {
    for (const m of trash) { try { m.delete(); } catch (e) { /* */ } }
  }
}
