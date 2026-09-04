/* WJM vision worker (classic worker).
 *
 * OpenCV.js (WASM) + ArUco + all the *geometric* extraction run here, so the
 * camera preview and overlay on the main thread never stutter. Text recognition
 * (Tesseract.js) runs on the main thread — it manages its own worker and needs a
 * canvas it can't get here.
 *
 * Protocol
 *   main → { type:"init", opencvUrls:[...] }
 *   ←    { type:"ready", hasAruco } | { type:"error", message }
 *   main → { type:"detect",  seq, width, height, buffer }         // live frame, RGBA transferred
 *   ←    { type:"markers",  seq, markers }
 *   main → { type:"analyze", seq, frames:[{buffer,width,height}], markers, mode } // burst, RGBA transferred
 *   ←    { type:"progress", seq, stage }
 *   ←    { type:"analyzed", seq, geometry, rectified:{buffer,width,height} }
 *   ←    { type:"analyzed", seq, error }
 *
 * SCANNER.md phase A: the burst is scored frame-by-frame (Tenengrad), the
 * sharpest is kept and rectified into a fixed 8.5x11 @ 300 DPI canvas. There is
 * no accept/reject gate — we always return the best of what we got; the review
 * screen reports how sharp it is.
 *
 * The chain mirrors wingjournal.pipeline.ingest_image from perspective
 * normalization onward (markers already give the page frame + orientation):
 *   rectify → literal-region detect + mask → metadata-block detect
 *   → text-line segmentation.  OCR + element parsing happen on the main thread.
 */
"use strict";

let cv = null;
let V = null;   // WJMVision
let detector = null;

// the fixed rectified canvas: 8.5 x 11 in at 300 DPI (SCANNER.md)
const LETTER_PX = [2550, 3300];

self.onmessage = (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") return void init(msg);
    if (msg.type === "detect") return void onDetect(msg);
    if (msg.type === "analyze") return void onAnalyze(msg);
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

function onAnalyze({ seq, frames, markers, mode }) {
  const progress = (stage) => post({ type: "progress", seq, stage });
  const trash = [];
  const keep = (m) => { trash.push(m); return m; };
  try {
    if (!frames || !frames.length) {
      return post({ type: "analyzed", seq, error: "no frames in the burst" });
    }

    // --- pick the sharpest frame of the burst (Tenengrad), drop the rest ---
    progress("choosing the sharpest frame");
    const scores = [];
    let best = null; // { src, gray, score, index }
    frames.forEach((f, i) => {
      const src = matFromBuffer(f.buffer, f.width, f.height);
      const gray = grayOf(src);
      const score = V.tenengrad(cv, gray);
      scores.push(Math.round(score * 100) / 100);
      if (!best || score > best.score) {
        if (best) { best.src.delete(); best.gray.delete(); }
        best = { src, gray, score, index: i };
      } else {
        src.delete(); gray.delete();
      }
    });
    const src = keep(best.src);
    const graySrc = keep(best.gray);
    const fw = frames[best.index].width;
    const fh = frames[best.index].height;

    // --- page frame from the winning frame's own fiducials ---
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
    // fixed 8.5x11 @ 300 DPI canvas; rectify picks INTER_AREA / LINEAR by ratio
    const { mat: normalized, width: nw, height: nh } =
      V.rectify(cv, src, quad, null, LETTER_PX);
    keep(normalized);

    const grayN = keep(grayOf(normalized));

    // page-size estimate from the sticker scale on the de-warped page (spec §11.2)
    let pageSize = null;
    if (stickerMode) {
      const nstk = detector.detect(grayN).filter((m) => m.id === V.CORNER_STICKER_ID);
      pageSize = V.estimatePageSize(V.detectCornerStickers(cv, grayN, nstk));
    }

    // markers re-found in normalized coords: block search + sharpness probes.
    // the winning frame's fiducials, in source pixels, for the report
    const shotFiducials = (stickerMode
      ? all.filter((m) => m.id === V.CORNER_STICKER_ID)
      : sheet
    ).map((m) => ({
      id: m.id, role: m.role || null,
      corners: m.corners.map(([x, y]) => [round3(x), round3(y)]),
      center: (m.center || [0, 0]).map(round3),
    }));
    const normMarkers = detector.detect(grayN);
    const markerBoxes = normMarkers.map((m) => {
      const xs = m.corners.map((p) => p[0]);
      const ys = m.corners.map((p) => p[1]);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
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
    // informational only in phase A — the burst already picked the best frame;
    // this tells the review screen how sharp that best is (SCANNER.md)
    const sharpness = V.assessSharpness(cv, grayN, normMarkers, regMarks);
    sharpness.focus_score = Math.round(best.score * 100) / 100;

    progress("segmenting text");
    const skipTop = block ? (block.bbox[1] + block.bbox[3]) / nh : 0;
    const lineBoxes = V.segmentLines(cv, grayMasked).filter(([, y]) => y >= skipTop * nh);

    const rectBuf = new Uint8ClampedArray(normalized.data);
    const geometry = {
      dictionary: "DICT_4X4_50",
      fiducial_mode: stickerMode ? "corner_stickers" : "printed_sheet",
      source: { width: fw, height: fh },
      burst: {
        count: frames.length, scores, winner: best.index,
        focus: sharpness.focus_score,
      },
      page_frame_quad: quad.map(([x, y]) => [round3(x), round3(y)]),
      orientation: { method: stickerMode ? "geometry" : "aruco_ids", degrees: 0 },
      normalized: { width: nw, height: nh },
      page_size_estimate: pageSize,
      detected_fiducials: shotFiducials,
      metadata_block: block,
      literal_assets: literals,
      line_boxes: lineBoxes,
      sharpness,
    };
    post(
      { type: "analyzed", seq, geometry, rectified: { buffer: rectBuf.buffer, width: nw, height: nh } },
      [rectBuf.buffer],
    );
  } catch (err) {
    post({ type: "analyzed", seq, error: String((err && err.stack) || err) });
  } finally {
    for (const m of trash) { try { m.delete(); } catch (e) { /* */ } }
  }
}
