/* WJM vision core — runs inside the worker (classic script, no ES modules so it
 * can sit alongside importScripts('opencv.js')).
 *
 * Mirrors wingjournal.vision.{aruco,boundary,rectify} so a capture here lines up
 * with `wingjournal ingest`:
 *   dictionary  DICT_4X4_50
 *   ids         0/1/2/3 = TOP_LEFT / TOP_RIGHT / BOTTOM_RIGHT / BOTTOM_LEFT
 *   page frame  outer corner of each of the 4 markers, ordered TL,TR,BR,BL
 *   normalized  longer side = rectify()'s targetLong (default 1600), aspect
 *               from the quad clamped to [1.15, 1.6]
 */
(function (root) {
  "use strict";

  const ROLE_BY_ID = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"];
  const ROLE_ORDER = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"];
  const TARGET_LONG_PX = 1600;
  const ASPECT_RANGE = [1.15, 1.6];

  // adhesive corner sticker (spec §11.2) — mirrors vision/aruco.py
  const CORNER_STICKER_ID = 10;
  const CORNER_STICKER_ARUCO_MM = 14.0;
  const PAPERS_MM = { letter: [215.9, 279.4], a4: [210.0, 297.0], legal: [215.9, 355.6] };

  // per-field metadata anchors (spec §11.3) — mirrors vision/aruco.py
  const FIELD_BY_MARKER_ID = {
    20: "document_id", 21: "page_id", 22: "topic_tags",
    23: "left", 24: "above", 25: "below", 26: "right",
  };
  const META_ROW1 = ["document_id", "page_id", "topic_tags"];
  const META_ROW2 = ["left", "above", "below", "right"];

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  function hasAruco(cv) {
    return (
      typeof cv.getPredefinedDictionary === "function" &&
      typeof cv.aruco_ArucoDetector === "function" &&
      typeof cv.aruco_DetectorParameters === "function"
    );
  }

  class MarkerDetector {
    constructor(cv) {
      this.cv = cv;
      this.dictionary = cv.getPredefinedDictionary(
        typeof cv.DICT_4X4_50 === "number" ? cv.DICT_4X4_50 : 0,
      );
      this.params = new cv.aruco_DetectorParameters();
      try {
        this.params.cornerRefinementMethod =
          typeof cv.CORNER_REFINE_SUBPIX === "number" ? cv.CORNER_REFINE_SUBPIX : 1;
      } catch (e) { /* read-only in some builds */ }

      let refine = null;
      try { refine = new cv.aruco_RefineParameters(10, 3, true); } catch (e) { /* older */ }
      try {
        this.detector = refine
          ? new cv.aruco_ArucoDetector(this.dictionary, this.params, refine)
          : new cv.aruco_ArucoDetector(this.dictionary, this.params);
      } catch (e) {
        this.detector = new cv.aruco_ArucoDetector(this.dictionary, this.params);
      }
    }

    /** @param {cv.Mat} grayMat single channel → marker list in grayMat pixels */
    detect(grayMat) {
      const cv = this.cv;
      const corners = new cv.MatVector();
      const ids = new cv.Mat();
      this.detector.detectMarkers(grayMat, corners, ids);

      const out = [];
      for (let i = 0; i < ids.rows; i++) {
        const id = ids.intAt(i, 0);
        const m = corners.get(i);
        const d = m.data32F; // x0,y0,x1,y1,x2,y2,x3,y3  (marker-local TL,TR,BR,BL)
        const pts = [[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]];
        m.delete();
        out.push({
          id,
          role: ROLE_BY_ID[id] ?? null,
          corners: pts,
          center: [
            (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4,
          ],
        });
      }
      corners.delete();
      ids.delete();
      out.sort((a, b) => a.id - b.id);
      return out;
    }
  }

  // wingjournal.vision.boundary.order_points
  function orderPoints(pts) {
    const w = pts.map((p) => ({ p, s: p[0] + p[1], d: p[1] - p[0] }));
    const pick = (k, max) =>
      w.reduce((a, b) => ((max ? b[k] > a[k] : b[k] < a[k]) ? b : a)).p;
    return [pick("s", false), pick("d", false), pick("s", true), pick("d", true)];
  }

  // wingjournal.vision.boundary.outer_corner_of
  function outerCorner(marker, centroid) {
    let best = marker.corners[0];
    let bestD = -1;
    for (const c of marker.corners) {
      const dd = dist(c, centroid);
      if (dd > bestD) { bestD = dd; best = c; }
    }
    return best;
  }

  /** Page quad (TL,TR,BR,BL) from the outer corner of each of the 4 markers. */
  function pageQuadFromMarkers(markers) {
    const byRole = {};
    for (const m of markers) if (m.role && !(m.role in byRole)) byRole[m.role] = m;
    if (ROLE_ORDER.some((r) => !byRole[r])) return null;

    const centers = ROLE_ORDER.map((r) => byRole[r].center);
    const centroid = [
      centers.reduce((s, c) => s + c[0], 0) / 4,
      centers.reduce((s, c) => s + c[1], 0) / 4,
    ];
    return orderPoints(ROLE_ORDER.map((r) => outerCorner(byRole[r], centroid)));
  }

  // wingjournal.vision.rectify.output_size
  function outputSize(quad, targetLong) {
    targetLong = targetLong || TARGET_LONG_PX;
    targetLong = Math.max(1, Math.round(targetLong));
    const [tl, tr, br, bl] = quad;
    const width = Math.max(dist(tl, tr), dist(bl, br));
    const height = Math.max(dist(tl, bl), dist(tr, br));
    if (Math.min(width, height) < 1e-3) return [targetLong, targetLong];
    const aspect = clamp(
      Math.max(width, height) / Math.min(width, height),
      ASPECT_RANGE[0], ASPECT_RANGE[1],
    );
    const shortPx = Math.max(1, Math.round(targetLong / aspect));
    return height >= width ? [shortPx, targetLong] : [targetLong, shortPx];
  }

  /**
   * Perspective-normalize the page.
   * @returns {{mat: cv.Mat, width:number, height:number}} caller deletes mat
   */
  function rectify(cv, srcMat, quad, targetLong, fixedSize) {
    const [w, h] = fixedSize
      ? [Math.max(1, Math.round(fixedSize[0])), Math.max(1, Math.round(fixedSize[1]))]
      : outputSize(quad, targetLong);
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, quad.flat());
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1]);
    const Hm = cv.getPerspectiveTransform(src, dst);
    // shrink -> INTER_AREA (box filter, no aliasing); near 1:1 or upscale ->
    // INTER_LINEAR. Never INTER_CUBIC: its negative lobes ring hard ink edges
    // and the downstream threshold / OCR then mangles the overshoot ("chunky").
    const srcLong = Math.max(
      dist(quad[0], quad[1]), dist(quad[1], quad[2]),
      dist(quad[2], quad[3]), dist(quad[3], quad[0]),
    );
    const AREA = typeof cv.INTER_AREA === "number" ? cv.INTER_AREA : 3;
    const LIN = typeof cv.INTER_LINEAR === "number" ? cv.INTER_LINEAR : 1;
    const out = new cv.Mat();
    cv.warpPerspective(
      srcMat, out, Hm, new cv.Size(w, h),
      Math.max(w, h) < 0.98 * srcLong ? AREA : LIN,
    );
    src.delete(); dst.delete(); Hm.delete();
    return { mat: out, width: w, height: h };
  }

  // Tenengrad focus score: mean Sobel gradient magnitude over `gray` (optionally
  // a [x,y,w,h] roi). Higher = sharper. Only meaningful when ranking frames of
  // the same scene against each other.
  function tenengrad(cv, gray, roi) {
    let g = gray;
    let owns = false;
    if (roi) { g = gray.roi(new cv.Rect(roi[0], roi[1], roi[2], roi[3])); owns = true; }
    const gx = new cv.Mat();
    const gy = new cv.Mat();
    const mag = new cv.Mat();
    cv.Sobel(g, gx, cv.CV_32F, 1, 0, 3);
    cv.Sobel(g, gy, cv.CV_32F, 0, 1, 3);
    cv.magnitude(gx, gy, mag);
    const s = cv.mean(mag)[0];
    gx.delete(); gy.delete(); mag.delete();
    if (owns) g.delete();
    return s;
  }

  // ===================================================================
  //  Close-up compositing (SCANNER.md phase B)
  //  Register a closer re-shot of one region into the fixed rectified canvas
  //  and paste it in, so that region gets real resolution.
  // ===================================================================

  const _hasORB = (cv) =>
    typeof cv.ORB === "function" && typeof cv.BFMatcher === "function" &&
    typeof cv.findHomography === "function";

  // clamp/validate a 3x3 homography: reject wild scale, shear or projection.
  function _saneHomography(cv, H) {
    if (!H || H.rows !== 3 || H.cols !== 3) return false;
    const d = H.data64F;
    const sx = Math.hypot(d[0], d[3]);
    const sy = Math.hypot(d[1], d[4]);
    if (!(sx > 0.15 && sx < 8 && sy > 0.15 && sy < 8)) return false;
    if (Math.abs(d[6]) > 0.004 || Math.abs(d[7]) > 0.004) return false; // perspective
    return true;
  }

  // Homography closeup-px -> canvas-px from ArUco ids the close-up shares with
  // the base. `baseCorners` maps id -> [[x,y]*4] in canvas space. Needs >= 2 ids
  // (>= 8 correspondences) — SCANNER.md's "keep two markers in view".
  function anchorHomography(cv, closeMarkers, baseCorners) {
    const srcPts = [];
    const dstPts = [];
    let shared = 0;
    for (const m of closeMarkers || []) {
      const base = baseCorners[m.id];
      if (!base) continue;
      shared += 1;
      for (let i = 0; i < 4; i++) {
        srcPts.push(m.corners[i][0], m.corners[i][1]);
        dstPts.push(base[i][0], base[i][1]);
      }
    }
    if (shared < 2) return { H: null, method: "anchors", shared };
    const s = cv.matFromArray(srcPts.length / 2, 1, cv.CV_32FC2, srcPts);
    const d = cv.matFromArray(dstPts.length / 2, 1, cv.CV_32FC2, dstPts);
    const H = shared >= 3
      ? cv.findHomography(s, d, cv.RANSAC, 4)
      : cv.getPerspectiveTransform(s, d);
    s.delete(); d.delete();
    if (!_saneHomography(cv, H)) { H.delete(); return { H: null, method: "anchors", shared }; }
    return { H, method: "anchors", shared };
  }

  const _hasAKAZE = (cv) => typeof cv.AKAZE === "function";

  // A mask (8-bit, same size as a `rect`-sized crop) that's white only over
  // known content — text lines + metadata cells from the base capture — so
  // feature detection lands on real landmarks (glyph strokes/corners) instead
  // of blank paper texture or compression noise. `rects` are canvas-space
  // [x,y,w,h]; `cropX/cropY` is the crop's canvas-space origin.
  function landmarkMask(cv, rows, cols, rects, cropX, cropY) {
    const m = cv.Mat.zeros(rows, cols, cv.CV_8UC1);
    let any = false;
    for (const r of rects || []) {
      const x0 = Math.max(0, Math.round(r[0] - cropX));
      const y0 = Math.max(0, Math.round(r[1] - cropY));
      const x1 = Math.min(cols, Math.round(r[0] + r[2] - cropX));
      const y1 = Math.min(rows, Math.round(r[1] + r[3] - cropY));
      if (x1 - x0 < 2 || y1 - y0 < 2) continue;
      cv.rectangle(m, new cv.Point(x0, y0), new cv.Point(x1, y1), new cv.Scalar(255), -1);
      any = true;
    }
    // an empty/blank mask would find nothing everywhere; fall back to "search
    // the whole crop" rather than starve the detector
    if (!any) m.setTo(new cv.Scalar(255));
    return m;
  }

  function _detectCompute(cv, detector, gray, mask) {
    const kp = new cv.KeyPointVector();
    const desc = new cv.Mat();
    detector.detectAndCompute(gray, mask || new cv.Mat(), kp, desc);
    return { kp, desc };
  }

  // Ratio-test match (Lowe's test — robust regardless of descriptor length,
  // unlike a fixed Hamming-distance cutoff) + RANSAC homography. `ORB` and
  // `AKAZE` keypoints both carry a computed dominant orientation and their
  // descriptors (rotated BRIEF / MLDB) are sampled relative to it, so this
  // matches correctly however the phone was rotated for the close-up —
  // in-plane rotation is exactly what these two detectors are built to
  // tolerate; no extra rotation search is needed.
  function _matchAndHomography(cv, k1, d1, k2, d2, offsetX, offsetY, minInliers, method) {
    let out = { H: null, method, inliers: 0 };
    if (d1.rows < 2 || d2.rows < 2) return out;
    const bf = new cv.BFMatcher(cv.NORM_HAMMING);
    const knn = new cv.DMatchVectorVector();
    bf.knnMatch(d1, d2, knn, 2);
    const sp = [];
    const dp = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() < 2) continue;
      const m = pair.get(0);
      const n = pair.get(1);
      if (m.distance < 0.75 * n.distance) {
        const p1 = k1.get(m.queryIdx).pt;
        const p2 = k2.get(m.trainIdx).pt;
        sp.push(p1.x, p1.y);
        dp.push(p2.x + offsetX, p2.y + offsetY);
      }
    }
    bf.delete(); knn.delete();
    if (sp.length / 2 >= minInliers) {
      const s = cv.matFromArray(sp.length / 2, 1, cv.CV_32FC2, sp);
      const d = cv.matFromArray(dp.length / 2, 1, cv.CV_32FC2, dp);
      const mask = new cv.Mat();
      const H = cv.findHomography(s, d, cv.RANSAC, 5, mask);
      const inliers = cv.countNonZero(mask);
      s.delete(); d.delete(); mask.delete();
      if (inliers >= minInliers && _saneHomography(cv, H)) out = { H, method, inliers };
      else H.delete();
    }
    return out;
  }

  // Feature-match the close-up (gray) against the canvas crop at `rect`.
  // `landmarks` (optional, canvas-space rects) restrict where keypoints are
  // sought on the canvas side to known text/content. Returns a
  // closeup-px -> canvas-px homography or null.
  function _featureHomography(cv, method, closeGray, canvasGray, rect, landmarks) {
    const [rx, ry, rw, rh] = rect.map(Math.round);
    const cx0 = Math.max(0, rx);
    const cy0 = Math.max(0, ry);
    const cw = Math.min(canvasGray.cols - cx0, rw);
    const ch = Math.min(canvasGray.rows - cy0, rh);
    if (cw < 8 || ch < 8) return { H: null, method, inliers: 0 };
    const crop = canvasGray.roi(new cv.Rect(cx0, cy0, cw, ch));
    const mask = landmarks ? landmarkMask(cv, ch, cw, landmarks, cx0, cy0) : null;

    const detector = method === "akaze" ? new cv.AKAZE() : new cv.ORB(1800);
    const a = _detectCompute(cv, detector, closeGray, null);
    const b = _detectCompute(cv, detector, crop, mask);
    const out = _matchAndHomography(cv, a.kp, a.desc, b.kp, b.desc, cx0, cy0, 8, method);

    detector.delete();
    a.kp.delete(); a.desc.delete(); b.kp.delete(); b.desc.delete();
    crop.delete();
    if (mask) mask.delete();
    return out;
  }

  // AKAZE: nonlinear-diffusion scale space handles blur/soft focus (a phone
  // close-up) better than ORB's pyramid in practice; try it first.
  function akazeHomography(cv, closeGray, canvasGray, rect, landmarks) {
    if (!_hasAKAZE(cv)) return { H: null, method: "akaze", inliers: 0 };
    return _featureHomography(cv, "akaze", closeGray, canvasGray, rect, landmarks);
  }

  // ORB fallback: faster, more keypoints — a second chance when AKAZE's
  // stricter response threshold finds too little on very sparse content.
  function orbHomography(cv, closeGray, canvasGray, rect, landmarks) {
    if (!_hasORB(cv)) return { H: null, method: "orb", inliers: 0 };
    return _featureHomography(cv, "orb", closeGray, canvasGray, rect, landmarks);
  }

  // ---- target planning (SCANNER.md PLAN_TARGETS) --------------------
  const _area = (r) => r[2] * r[3];
  const _pad = (r, px, W, H) => {
    const x = Math.max(0, r[0] - px);
    const y = Math.max(0, r[1] - px);
    return [x, y, Math.min(W - x, r[2] + 2 * px), Math.min(H - y, r[3] + 2 * px)];
  };
  const _overlap = (a, b) =>
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];
  const _union = (a, b) => {
    const x = Math.min(a[0], b[0]);
    const y = Math.min(a[1], b[1]);
    return [x, y, Math.max(a[0] + a[2], b[0] + b[2]) - x, Math.max(a[1] + a[3], b[1] + b[3]) - y];
  };

  /**
   * Up to 5 canvas-space rects that carry detail worth a closer shot: the
   * metadata block, any literal-image regions, and clusters of body text lines.
   * (SCANNER.md — "aruco / box clues".)
   */
  function planTargets(canvasW, canvasH, block, literals, lineBoxes) {
    let rects = [];
    if (block && block.bbox) rects.push(_pad(block.bbox, canvasH * 0.012, canvasW, canvasH));
    for (const lit of literals || []) {
      if (lit && lit.bbox) rects.push(_pad(lit.bbox, canvasH * 0.01, canvasW, canvasH));
    }
    // cluster body lines into paragraph-ish blocks (gap < ~4 line-heights); a
    // block is only worth a close-up if it spans a few lines / enough height
    const lines = (lineBoxes || []).slice().sort((a, b) => a[1] - b[1]);
    let cur = null;
    let n = 0;
    const flush = () => {
      if (cur && (n >= 3 || cur[3] >= canvasH * 0.05)) {
        rects.push(_pad(cur, canvasH * 0.008, canvasW, canvasH));
      }
    };
    for (const lb of lines) {
      if (cur && lb[1] - (cur[1] + cur[3]) < 4 * lb[3]) { cur = _union(cur, lb); n += 1; }
      else { flush(); cur = lb.slice(); n = 1; }
    }
    flush();

    // merge overlaps, drop specks, clamp each to <= half width / 40% height so a
    // closer shot actually gains resolution
    const merged = [];
    for (const r of rects) {
      const hit = merged.find((m) => _overlap(m, r));
      if (hit) Object.assign(hit, _union(hit, r));
      else merged.push(r.slice());
    }
    const minArea = 0.012 * canvasW * canvasH;
    const maxW = canvasW * 0.55;
    const maxH = canvasH * 0.42;
    const out = merged.map((r) => {
      if (r[2] <= maxW && r[3] <= maxH) return r;
      const cx = r[0] + r[2] / 2;
      const cy = r[1] + r[3] / 2;
      const w = Math.min(r[2], maxW);
      const h = Math.min(r[3], maxH);
      return [
        Math.max(0, Math.min(canvasW - w, cx - w / 2)),
        Math.max(0, Math.min(canvasH - h, cy - h / 2)),
        w, h,
      ];
    }).filter((r) => _area(r) >= minArea);
    out.sort((a, b) => _area(b) - _area(a));
    return out.slice(0, 5).map((r) => r.map((v) => Math.round(v)));
  }

  // Warp `closeSrc` (RGBA) through H into canvas space and alpha-feather it over
  // `canvas` (RGBA), clipped to `rect` [x,y,w,h]. Mutates `canvas`.
  //
  // Everything happens in a small buffer sized to `rect` (+ feather padding),
  // not the full canvas: the original version warped and float-blended at the
  // *whole 2550x3300 canvas* for every close-up (four ~135 MB float32 buffers
  // per call), which is fine on a dev machine but starves or hangs a phone's
  // WASM heap — the close-up pass would silently stop responding. `H` is
  // translated into the ROI's local coordinates so the same warp still lands
  // in the same place.
  function compositeInto(cv, canvas, closeSrc, H, rect, feather = 14) {
    const pad = Math.ceil(feather * 2);
    const rx = Math.max(0, Math.round(rect[0]));
    const ry = Math.max(0, Math.round(rect[1]));
    const rw0 = Math.round(rect[2]);
    const rh0 = Math.round(rect[3]);
    const bx0 = Math.max(0, rx - pad);
    const by0 = Math.max(0, ry - pad);
    const bx1 = Math.min(canvas.cols, rx + rw0 + pad);
    const by1 = Math.min(canvas.rows, ry + rh0 + pad);
    const bw = Math.max(1, bx1 - bx0);
    const bh = Math.max(1, by1 - by0);

    // H with the ROI's top-left subtracted out (translate-then-H, expanded
    // algebraically so no extra matrix-multiply API is required)
    const h = H.data64F;
    const Hroi = cv.matFromArray(3, 3, cv.CV_64F, [
      h[0] - bx0 * h[6], h[1] - bx0 * h[7], h[2] - bx0 * h[8],
      h[3] - by0 * h[6], h[4] - by0 * h[7], h[5] - by0 * h[8],
      h[6], h[7], h[8],
    ]);

    const warped = new cv.Mat();
    cv.warpPerspective(
      closeSrc, warped, Hroi, new cv.Size(bw, bh),
      cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0),
    );
    Hroi.delete();

    // feathered rectangular mask, in ROI-local coords
    const mask = cv.Mat.zeros(bh, bw, cv.CV_8UC1);
    const mx0 = Math.max(0, rx - bx0 + feather);
    const my0 = Math.max(0, ry - by0 + feather);
    const mx1 = Math.min(bw, rx - bx0 + rw0 - feather);
    const my1 = Math.min(bh, ry - by0 + rh0 - feather);
    cv.rectangle(mask, new cv.Point(mx0, my0), new cv.Point(mx1, my1), new cv.Scalar(255), -1);
    if (feather > 0) cv.GaussianBlur(mask, mask, new cv.Size(0, 0), feather / 2);
    // also mask out where the warp produced nothing (alpha 0)
    const chans = new cv.MatVector();
    cv.split(warped, chans);
    const alpha = chans.get(3);
    cv.min(mask, alpha, mask);
    chans.delete(); alpha.delete();

    const m3 = new cv.Mat();
    cv.cvtColor(mask, m3, cv.COLOR_GRAY2RGBA);
    m3.convertTo(m3, cv.CV_32FC4, 1 / 255);
    const roi = canvas.roi(new cv.Rect(bx0, by0, bw, bh));
    const fg = new cv.Mat();
    const bg = new cv.Mat();
    warped.convertTo(fg, cv.CV_32FC4);
    roi.convertTo(bg, cv.CV_32FC4);
    cv.multiply(fg, m3, fg);
    const inv = new cv.Mat();
    cv.subtract(new cv.Mat(bh, bw, cv.CV_32FC4, new cv.Scalar(1, 1, 1, 1)), m3, inv);
    cv.multiply(bg, inv, bg);
    cv.add(fg, bg, bg);
    bg.convertTo(roi, cv.CV_8UC4);
    roi.delete();
    warped.delete(); mask.delete(); m3.delete(); fg.delete(); bg.delete(); inv.delete();
  }

  // ===================================================================
  //  Geometric recognition — ports of
  //    src/wingjournal/recognition/metadata_block.py
  //    src/wingjournal/recognition/text/segment.py
  //    src/wingjournal/vision/literal_box.py
  //  Every function takes an already-grayscale single-channel cv.Mat.
  // ===================================================================

  function adaptiveInv(C, gray) {
    const b = new C.Mat();
    C.adaptiveThreshold(gray, b, 255, C.ADAPTIVE_THRESH_GAUSSIAN_C, C.THRESH_BINARY_INV, 35, 11);
    return b;
  }

  /** column sums (dim 0) or row sums (dim 1) over a rect, as a plain Array of counts */
  function projection(C, mask, rect, dim) {
    const roi = mask.roi(rect);
    const out = new C.Mat();
    const REDUCE_SUM = typeof C.REDUCE_SUM === "number" ? C.REDUCE_SUM : 0;
    C.reduce(roi, out, dim, REDUCE_SUM, C.CV_32S);
    const arr = Array.from(out.data32S, (v) => v / 255); // mask is 0/255 -> counts
    roi.delete();
    out.delete();
    return arr;
  }

  // metadata_block._line_mask
  function lineMask(C, binary, horizontal, scale = 20) {
    const len = Math.max(8, Math.floor((horizontal ? binary.cols : binary.rows) / scale));
    const kernel = C.getStructuringElement(
      C.MORPH_RECT, horizontal ? new C.Size(len, 1) : new C.Size(1, len),
    );
    const eroded = new C.Mat();
    const out = new C.Mat();
    C.erode(binary, eroded, kernel);
    C.dilate(eroded, out, kernel);
    kernel.delete();
    eroded.delete();
    return out;
  }

  // metadata_block._positions — centres of the "on" runs of a profile
  function profilePositions(profile, minFrac = 0.4) {
    const peak = Math.max(0, ...profile);
    const thr = peak ? peak * minFrac : 1;
    const on = profile.map((v) => v > thr);
    const runs = [];
    let start = null;
    for (let i = 0; i < on.length; i++) {
      if (on[i] && start === null) start = i;
      else if (!on[i] && start !== null) { runs.push((start + i - 1) >> 1); start = null; }
    }
    if (start !== null) runs.push((start + on.length - 1) >> 1);
    return runs;
  }

  // metadata_block._row_cells
  function rowCells(C, vertical, bx, bw, y0, y1, minCell) {
    const prof = projection(C, vertical, new C.Rect(bx, y0, bw, y1 - y0), 0);
    const set = new Set([bx, bx + bw]);
    for (const p of profilePositions(prof)) set.add(bx + p);
    const xs = [...set].sort((a, b) => a - b);
    const cells = [];
    for (let i = 0; i < xs.length - 1; i++) {
      const w = xs[i + 1] - xs[i];
      if (w >= minCell) cells.push([xs[i], y0, w, y1 - y0]);
    }
    return cells;
  }

  // ---- registration marks + sharpness (registration.py / sharpness.py) ----

  // registration._edge_acutance — 0 (mush) .. 1 (crisp)
  function edgeAcutance(gray, cx, cy, size) {
    const h = gray.rows;
    const w = gray.cols;
    const half = Math.max(4, Math.floor(size * 0.7));
    const data = gray.data;
    const ix = Math.min(w - 1, Math.max(0, Math.round(cx)));
    const iy = Math.min(h - 1, Math.max(0, Math.round(cy)));
    const lines = [];
    { // horizontal
      const x0 = Math.max(0, ix - half);
      const x1 = Math.min(w, ix + half);
      const p = [];
      for (let x = x0; x < x1; x++) p.push(data[iy * w + x]);
      lines.push(p);
    }
    { // vertical
      const y0 = Math.max(0, iy - half);
      const y1 = Math.min(h, iy + half);
      const p = [];
      for (let y = y0; y < y1; y++) p.push(data[y * w + ix]);
      lines.push(p);
    }
    let best = 0;
    for (const p of lines) {
      if (p.length < 6) continue;
      const span = Math.max(...p) - Math.min(...p);
      if (span < 30) continue;
      let g = 0;
      for (let i = 1; i < p.length; i++) g = Math.max(g, Math.abs(p[i] - p[i - 1]));
      best = Math.max(best, g / span);
    }
    return Math.min(1, Math.max(0, best));
  }

  function squareish(C, cnt) {
    const area = C.contourArea(cnt);
    if (area < 16) return null;
    const r = C.boundingRect(cnt);
    if (Math.min(r.width, r.height) === 0) return null;
    if (Math.abs(r.width - r.height) > 0.35 * Math.max(r.width, r.height)) return null;
    if (area / (r.width * r.height) < 0.6) return null;
    return { side: Math.max(r.width, r.height), cx: r.x + r.width / 2, cy: r.y + r.height / 2 };
  }

  /**
   * detect_registration_marks — concentric-square marks.
   * @returns {{center:number[], size:number, acutance:number, rings:number}[]}
   */
  function detectRegistrationMarks(C, gray, roi, exclude) {
    let g = gray;
    let ox = 0;
    let oy = 0;
    let owns = false;
    if (roi) {
      g = gray.roi(new C.Rect(roi[0], roi[1], roi[2], roi[3]));
      ox = roi[0]; oy = roi[1]; owns = true;
    }
    const H = g.rows;
    const W = g.cols;
    if (H < 8 || W < 8) { if (owns) g.delete(); return []; }
    const lo = 0.006 * Math.max(H, W);
    const hi = 0.06 * Math.max(H, W);

    const blur = new C.Mat();
    const binary = new C.Mat();
    C.GaussianBlur(g, blur, new C.Size(3, 3), 0);
    C.threshold(blur, binary, 0, 255, C.THRESH_BINARY_INV | C.THRESH_OTSU);
    const contours = new C.MatVector();
    const hierarchy = new C.Mat();
    C.findContours(binary, contours, hierarchy, C.RETR_CCOMP, C.CHAIN_APPROX_SIMPLE);
    const hd = hierarchy.data32S; // [next, prev, child, parent] * n

    const excl = exclude || [];
    const hidden = (cx, cy) => excl.some(
      ([bx, by, bw, bh]) => cx + ox >= bx && cx + ox <= bx + bw && cy + oy >= by && cy + oy <= by + bh,
    );

    const found = [];
    for (let i = 0; i < contours.size(); i++) {
      if (hd[i * 4 + 3] !== -1) continue; // outermost only
      const cnt = contours.get(i);
      const outer = squareish(C, cnt);
      cnt.delete();
      if (!outer || outer.side < lo || outer.side > hi || hidden(outer.cx, outer.cy)) continue;

      const near = 0.25 * outer.side;
      let rings = 1;
      let child = hd[i * 4 + 2];
      while (child !== -1) {
        const cc = contours.get(child);
        const inner = squareish(C, cc);
        cc.delete();
        if (inner && inner.side >= 0.12 * outer.side && inner.side <= 0.72 * outer.side
          && Math.abs(inner.cx - outer.cx) < near && Math.abs(inner.cy - outer.cy) < near) rings += 1;
        child = hd[child * 4];
      }
      if (rings < 2) continue;
      found.push({
        center: [outer.cx + ox, outer.cy + oy],
        size: outer.side,
        acutance: Math.round(edgeAcutance(g, outer.cx, outer.cy, outer.side) * 1000) / 1000,
        rings: Math.min(rings, 3),
      });
    }

    found.sort((a, b) => (b.rings - a.rings) || (b.acutance - a.acutance) || (b.size - a.size));
    const out = [];
    for (const m of found) {
      const dup = out.some((d) =>
        Math.abs(m.center[0] - d.center[0]) < 0.5 * m.size
        && Math.abs(m.center[1] - d.center[1]) < 0.5 * m.size);
      if (!dup) out.push(m);
    }
    blur.delete(); binary.delete(); contours.delete(); hierarchy.delete();
    if (owns) g.delete();
    return out.slice(0, 4);
  }

  function marksToQuad(marks) {
    if (marks.length !== 4) return null;
    return orderPoints(marks.map((m) => m.center));
  }

  // sharpness.laplacian_variance
  function laplacianVariance(C, gray, roi) {
    let g = gray;
    let owns = false;
    if (roi) { g = gray.roi(new C.Rect(roi[0], roi[1], roi[2], roi[3])); owns = true; }
    const lap = new C.Mat();
    C.Laplacian(g, lap, C.CV_64F);
    const mean = new C.Mat();
    const std = new C.Mat();
    C.meanStdDev(lap, mean, std);
    const s = std.data64F[0];
    lap.delete(); mean.delete(); std.delete();
    if (owns) g.delete();
    return s * s;
  }

  const LAPVAR_FLOOR = 25;
  const LAPVAR_CEIL = 320;
  const MIN_ACUTANCE = 0.30;
  const MIN_SCORE = 0.45;

  /**
   * sharpness.assess — score a page at the known fiducials.
   * @param markers  [{id, corners}]   @param regMarks [{center, size, acutance}]
   */
  function assessSharpness(C, gray, markers, regMarks) {
    const lapvar = laplacianVariance(C, gray);
    const globalScore = clamp((lapvar - LAPVAR_FLOOR) / (LAPVAR_CEIL - LAPVAR_FLOOR), 0, 1);

    const probes = [];
    for (const m of markers || []) {
      const p = m.corners;
      const cx = (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4;
      const cy = (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4;
      const side = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
      const a = Math.round(edgeAcutance(gray, cx, cy, side) * 1000) / 1000;
      probes.push({ name: `marker:${m.id ?? "?"}`, acutance: a, sharp: a >= MIN_ACUTANCE });
    }
    (regMarks || []).forEach((rm, i) => {
      probes.push({ name: `registration:${i}`, acutance: rm.acutance, sharp: rm.acutance >= MIN_ACUTANCE });
    });

    const probeScore = probes.length
      ? probes.reduce((s, p) => s + p.acutance, 0) / probes.length : globalScore;
    const score = probes.length
      ? Math.round((0.7 * probeScore + 0.3 * globalScore) * 1000) / 1000
      : Math.round(globalScore * 1000) / 1000;
    const blurry = score < MIN_SCORE || probes.some((p) => !p.sharp);
    return {
      score,
      laplacian_variance: Math.round(lapvar * 10) / 10,
      global_score: Math.round(globalScore * 1000) / 1000,
      probe_score: Math.round(probeScore * 1000) / 1000,
      probes,
      blurry,
    };
  }

  // metadata_block._from_field_anchors (spec §11.3) — primary path. Each
  // field's box is the span from its anchor (ids 20-26) to the next.
  function anchorBBox(corners) {
    const xs = corners.map((p) => p[0]);
    const ys = corners.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }

  function metaFromFieldAnchors(markers) {
    const anchors = {};
    for (const m of markers || []) {
      const name = FIELD_BY_MARKER_ID[m.id];
      if (name) anchors[name] = anchorBBox(m.corners);
    }
    const names = Object.keys(anchors);
    if (names.length < 4) return null;

    const mid = (a) => (a[1] + a[3]) / 2;
    const cys = names.map((n) => mid(anchors[n])).sort((a, b) => a - b);
    const split = (cys[0] + cys[cys.length - 1]) / 2;
    const rowOf = {};
    for (const n of names) rowOf[n] = mid(anchors[n]) < split ? 0 : 1;

    const xs0 = Math.min(...names.map((n) => anchors[n][0]));
    const xs1 = Math.max(...names.map((n) => anchors[n][2]));
    const ys0 = Math.min(...names.map((n) => anchors[n][1]));
    const ys1 = Math.max(...names.map((n) => anchors[n][3]));

    const fieldCells = {};
    for (const [order, ridx] of [[META_ROW1, 0], [META_ROW2, 1]]) {
      const present = order
        .filter((n) => n in anchors && rowOf[n] === ridx)
        .map((n) => [n, anchors[n]])
        .sort((a, b) => a[1][0] - b[1][0]);
      for (let i = 0; i < present.length; i++) {
        const [name, a] = present[i];
        const [ax0, ay0, ax1, ay1] = a;
        const aw = ax1 - ax0;
        const ah = ay1 - ay0;
        const left = ax1 + 0.15 * aw;
        const right = i + 1 < present.length
          ? present[i + 1][1][0] - 0.15 * aw
          : xs1 + 0.06 * (xs1 - xs0);
        fieldCells[name] = [left, ay0 + 0.12 * ah, Math.max(1, right - left), ah];
      }
    }
    if (!Object.keys(fieldCells).length) return null;

    const rows = Object.values(rowOf);
    const bothRows = rows.includes(0) && rows.includes(1);
    return {
      bbox: [xs0, ys0, xs1 - xs0, ys1 - ys0],
      row_divider_y: bothRows ? (ys0 + ys1) / 2 : ys1,
      row1_cells: META_ROW1.filter((n) => n in fieldCells).map((n) => fieldCells[n]),
      row2_cells: META_ROW2.filter((n) => n in fieldCells).map((n) => fieldCells[n]),
      confidence: Math.round(Math.min(0.97, 0.6 + 0.05 * names.length) * 1000) / 1000,
      detection: "field_anchors",
      registration_marks: [],
      field_cells: fieldCells,
    };
  }

  // metadata_block._from_registration_marks
  function metaFromMarks(C, gray, searchFrac, markerBoxes) {
    const h = gray.rows;
    const w = gray.cols;
    const band = [0, 0, w, Math.max(1, Math.floor(h * searchFrac))];
    const marks = detectRegistrationMarks(C, gray, band, markerBoxes);
    const quad = marksToQuad(marks);
    if (!quad) return null;
    const xs = quad.map((p) => p[0]);
    const ys = quad.map((p) => p[1]);
    const bx = Math.min(...xs);
    const by = Math.min(...ys);
    const bw = Math.max(...xs) - bx;
    const bh = Math.max(...ys) - by;
    if (bw < 0.35 * w || bh < 6 || bw / Math.max(bh, 1) < 2.0) return null;
    const dividerY = by + bh / 2;
    const split = (n, y0, y1) => Array.from({ length: n }, (_, i) => [bx + bw * i / n, y0, bw / n, y1 - y0]);
    const acut = marks.reduce((s, m) => s + m.acutance, 0) / marks.length;
    return {
      bbox: [bx, by, bw, bh],
      row_divider_y: dividerY,
      row1_cells: split(3, by, dividerY),
      row2_cells: split(4, dividerY, by + bh),
      confidence: Math.round((0.75 + 0.25 * acut) * 1000) / 1000,
      detection: "registration_marks",
      registration_marks: marks.map((m) => [
        Math.round(m.center[0] * 10) / 10, Math.round(m.center[1] * 10) / 10,
        Math.round(m.size * 10) / 10, m.acutance,
      ]),
    };
  }

  /**
   * detect_metadata_block — registration marks first, ruled lines as fallback.
   * @returns {{bbox:number[], row_divider_y:number, row1_cells:number[][],
   *            row2_cells:number[][], confidence:number, detection:string,
   *            registration_marks:number[][]} | null}
   */
  function detectMetadataBlock(C, gray, searchFrac = 0.42, markerBoxes = null, markers = null) {
    const viaAnchors = metaFromFieldAnchors(markers);
    if (viaAnchors) return viaAnchors;

    const viaMarks = metaFromMarks(C, gray, searchFrac, markerBoxes);
    if (viaMarks) return viaMarks;

    const h = gray.rows;
    const w = gray.cols;
    const topH = Math.max(1, Math.floor(h * searchFrac));
    const top = gray.roi(new C.Rect(0, 0, w, topH));
    const binary = adaptiveInv(C, top);
    const horizontal = lineMask(C, binary, true);
    const vertical = lineMask(C, binary, false);
    const grid = new C.Mat();
    C.bitwise_or(horizontal, vertical, grid);

    const contours = new C.MatVector();
    const hierarchy = new C.Mat();
    C.findContours(grid, contours, hierarchy, C.RETR_EXTERNAL, C.CHAIN_APPROX_SIMPLE);

    let best = null;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const r = C.boundingRect(c);
      c.delete();
      if (r.width < 0.55 * w || r.height < 0.02 * h) continue;
      const ar = r.width / Math.max(r.height, 1);
      if (ar < 2.0 || ar > 24.0) continue;
      if (!best || r.width * r.height > best.width * best.height) best = r;
    }

    let result = null;
    if (best) {
      const { x: bx, y: by, width: bw, height: bh } = best;
      const m = Math.max(2, bh >> 3);
      const rowProfile = projection(C, horizontal, new C.Rect(bx, by + m, bw, Math.max(1, bh - 2 * m)), 1);
      const peak = Math.max(0, ...rowProfile);
      const dividerY = peak > 0
        ? by + m + rowProfile.indexOf(peak)
        : Math.round(by + bh / 2);

      const minCell = 0.04 * bw;
      const r1 = rowCells(C, vertical, bx, bw, by, dividerY, minCell);
      const r2 = rowCells(C, vertical, bx, bw, dividerY, by + bh, minCell);
      const cellScore = 0.5 * (r1.length === 3) + 0.5 * (r2.length === 4);
      const widthScore = Math.min(1.0, bw / (0.85 * w));
      result = {
        bbox: [bx, by, bw, bh],
        row_divider_y: dividerY,
        row1_cells: r1,
        row2_cells: r2,
        confidence: Math.round((0.4 + 0.4 * cellScore + 0.2 * widthScore) * 1000) / 1000,
        detection: "ruled_lines",
        registration_marks: [],
      };
    }

    top.delete(); binary.delete(); horizontal.delete(); vertical.delete();
    grid.delete(); contours.delete(); hierarchy.delete();
    return result;
  }

  // ---- segment.py ---------------------------------------------------

  // segment._runs — 1D run extraction with a gap tolerance
  function runs1d(on, minGap, minLen) {
    const out = [];
    let start = null;
    let gap = 0;
    for (let i = 0; i < on.length; i++) {
      if (on[i]) { if (start === null) start = i; gap = 0; }
      else if (start !== null) {
        gap += 1;
        if (gap > minGap) {
          if (i - gap - start >= minLen) out.push([start, i - gap]);
          start = null;
        }
      }
    }
    if (start !== null && on.length - start >= minLen) out.push([start, on.length]);
    return out;
  }

  /** segment_lines — text-line bounding boxes [x,y,w,h], top to bottom */
  function segmentLines(C, gray, minLineH = 6) {
    const binary = adaptiveInv(C, gray);
    const h = binary.rows;
    const w = binary.cols;
    const rowInk = projection(C, binary, new C.Rect(0, 0, w, h), 1);
    const thr = Math.max(1, 0.01 * w);
    const bands = runs1d(rowInk.map((v) => v > thr), Math.max(2, Math.floor(h / 60)), minLineH);
    const out = [];
    for (const [y0, y1] of bands) {
      const cols = projection(C, binary, new C.Rect(0, y0, w, y1 - y0), 0);
      let x0 = -1;
      let x1 = -1;
      for (let i = 0; i < cols.length; i++) if (cols[i] > 0) { if (x0 < 0) x0 = i; x1 = i; }
      if (x0 < 0) continue;
      out.push([x0, y0, x1 - x0 + 1, y1 - y0]);
    }
    binary.delete();
    return out;
  }

  // ---- literal_box.py ---------------------------------------------

  function patchMean(data, cols, rows, cx, cy, half) {
    const x0 = Math.max(0, cx - half);
    const x1 = Math.min(cols, cx + half);
    const y0 = Math.max(0, cy - half);
    const y1 = Math.min(rows, cy + half);
    if (x1 <= x0 || y1 <= y0) return 0;
    let on = 0;
    let n = 0;
    for (let y = y0; y < y1; y++) {
      const base = y * cols;
      for (let x = x0; x < x1; x++) { if (data[base + x] > 0) on++; n++; }
    }
    return n ? on / n : 0;
  }

  function cornerScores(data, cols, rows, x, y, bw, bh, size) {
    const half = Math.max(2, size >> 1);
    const inward = Math.max(3, Math.floor(size * 0.8));
    const wedgeHalf = Math.max(2, Math.floor(size / 3));
    const corners = [
      [x, y, 1, 1], [x + bw, y, -1, 1], [x + bw, y + bh, -1, -1], [x, y + bh, 1, -1],
    ];
    let minTip = Infinity;
    let minWedge = Infinity;
    for (const [cx, cy, dx, dy] of corners) {
      minTip = Math.min(minTip, patchMean(data, cols, rows, cx + dx * half, cy + dy * half, half));
      minWedge = Math.min(minWedge, patchMean(data, cols, rows, cx + dx * inward, cy + dy * inward, wedgeHalf));
    }
    return [minTip, minWedge];
  }

  /**
   * detect_literal_assets — rectangles with four solid diagonal corner fills.
   * @returns {{bbox:number[], confidence:number}[]}
   */
  function detectLiteralAssets(C, gray, opts = {}) {
    const {
      minAreaFrac = 0.004, cornerFrac = 0.09,
      minTipFill = 0.5, minWedgeFill = 0.3, maxEdgeFill = 0.3,
    } = opts;
    const h = gray.rows;
    const w = gray.cols;

    const edges = adaptiveInv(C, gray);
    const binary = new C.Mat();
    C.threshold(gray, binary, 0, 255, C.THRESH_BINARY_INV | C.THRESH_OTSU);
    const bdata = binary.data;

    const contours = new C.MatVector();
    const hierarchy = new C.Mat();
    C.findContours(edges, contours, hierarchy, C.RETR_LIST, C.CHAIN_APPROX_SIMPLE);

    const cand = [];
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      cand.push({ area: C.contourArea(c), rect: C.boundingRect(c) });
      c.delete();
    }
    cand.sort((a, b) => b.area - a.area);

    const out = [];
    const seen = [];
    for (const { area, rect } of cand) {
      if (area < minAreaFrac * h * w || area > 0.9 * h * w) continue;
      const { x, y, width: bw, height: bh } = rect;
      if (bw < 0.08 * w || bh < 0.08 * h) continue;
      if (seen.some(([sx, sy]) => Math.abs(x - sx) < 15 && Math.abs(y - sy) < 15)) continue;

      const size = Math.max(4, Math.floor(Math.min(bw, bh) * cornerFrac));
      const [tip, wedge] = cornerScores(bdata, w, h, x, y, bw, bh, size);
      if (tip < minTipFill || wedge < minWedgeFill) continue;
      const edge = Math.max(
        patchMean(bdata, w, h, x + (bw >> 1), y, size),
        patchMean(bdata, w, h, x + (bw >> 1), y + bh, size),
        patchMean(bdata, w, h, x, y + (bh >> 1), size),
        patchMean(bdata, w, h, x + bw, y + (bh >> 1), size),
      );
      if (edge > maxEdgeFill) continue;

      seen.push([x, y]);
      out.push({
        bbox: [x, y, bw, bh],
        confidence: Math.round(Math.min(1, 0.5 * tip + 0.5 * wedge) * (1 - edge) * 1000) / 1000,
      });
    }

    edges.delete(); binary.delete(); contours.delete(); hierarchy.delete();
    return out;
  }

  // ===================================================================
  //  Adhesive corner stickers — port of vision/corner_sticker.py
  // ===================================================================

  const STICKER_ROLE_BY_QUADRANT = {
    "-1,-1": "TOP_LEFT", "1,-1": "TOP_RIGHT", "1,1": "BOTTOM_RIGHT", "-1,1": "BOTTOM_LEFT",
  };

  const markerSide = (m) => {
    const c = m.corners;
    return (dist(c[0], c[1]) + dist(c[1], c[2]) + dist(c[2], c[3]) + dist(c[3], c[0])) / 4;
  };

  // boundary.complete_quad_from_three
  function completeQuadFromThree(byRole) {
    const have = Object.keys(byRole);
    if (have.length !== 3) return null;
    const missing = ROLE_ORDER.find((r) => !(r in byRole));
    const opp = {
      TOP_LEFT: ["TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"],
      TOP_RIGHT: ["TOP_LEFT", "BOTTOM_RIGHT", "BOTTOM_LEFT"],
      BOTTOM_RIGHT: ["TOP_RIGHT", "BOTTOM_LEFT", "TOP_LEFT"],
      BOTTOM_LEFT: ["TOP_LEFT", "BOTTOM_RIGHT", "TOP_RIGHT"],
    }[missing];
    const [a, b, c] = opp.map((r) => byRole[r]);
    const pts = Object.assign({}, byRole);
    pts[missing] = [a[0] + b[0] - c[0], a[1] + b[1] - c[1]];
    return orderPoints(ROLE_ORDER.map((r) => pts[r]));
  }

  // corner_sticker._find_bracket — wedge tip = the page corner; falls back to
  // the ArUco's own outer corner when the sticker was rotated the wrong way
  function findWedgeTip(C, gray, marker, outward) {
    const h = gray.rows;
    const w = gray.cols;
    const center = marker.center;
    const side = markerSide(marker);
    const proj1 = (p) => (p[0] - center[0]) * outward[0] + (p[1] - center[1]) * outward[1];
    const arucoOuter = marker.corners.reduce((a, b) => (proj1(b) > proj1(a) ? b : a));
    const reach = proj1(arucoOuter);
    const estimate = [arucoOuter[0] + outward[0] * 0.7 * side, arucoOuter[1] + outward[1] * 0.7 * side];

    const px = center[0] + outward[0] * side;
    const py = center[1] + outward[1] * side;
    const r = Math.floor(0.9 * side);
    const x0 = Math.max(0, Math.floor(px - r));
    const x1 = Math.min(w, Math.floor(px + r));
    const y0 = Math.max(0, Math.floor(py - r));
    const y1 = Math.min(h, Math.floor(py + r));
    if (x1 - x0 < 6 || y1 - y0 < 6) return [estimate, false];

    const roi = gray.roi(new C.Rect(x0, y0, x1 - x0, y1 - y0));
    const ink = new C.Mat();
    C.threshold(roi, ink, 0, 255, C.THRESH_BINARY_INV | C.THRESH_OTSU);
    const poly = C.matFromArray(4, 1, C.CV_32SC2,
      marker.corners.flatMap(([x, y]) => [Math.round(x - x0), Math.round(y - y0)]));
    const vec = new C.MatVector();
    vec.push_back(poly);
    C.fillPoly(ink, vec, new C.Scalar(0));
    const data = ink.data;
    const iw = ink.cols;
    let bestTip = null;
    let bestProj = -Infinity;
    let n = 0;
    for (let y = 0; y < ink.rows; y++) {
      for (let x = 0; x < iw; x++) {
        if (!data[y * iw + x]) continue;
        n++;
        const proj = (x + x0 - center[0]) * outward[0] + (y + y0 - center[1]) * outward[1];
        if (proj > bestProj) { bestProj = proj; bestTip = [x + x0, y + y0]; }
      }
    }
    roi.delete(); ink.delete(); poly.delete(); vec.delete();
    if (n < Math.max(20, 0.02 * (x1 - x0) * (y1 - y0))) return [estimate, false];
    if (bestProj < 1.1 * reach) return [estimate, false];
    return [bestTip, true];
  }

  /**
   * detect_corner_stickers.
   * @param stickerMarkers  markers already filtered to CORNER_STICKER_ID
   * @returns {{marker, outward:number[], corner_point:number[], bracket_found:bool, role:string}[]}
   */
  function detectCornerStickers(C, gray, stickerMarkers) {
    if (!stickerMarkers.length) return [];
    const cx = stickerMarkers.reduce((s, m) => s + m.center[0], 0) / stickerMarkers.length;
    const cy = stickerMarkers.reduce((s, m) => s + m.center[1], 0) / stickerMarkers.length;
    const out = [];
    for (const m of stickerMarkers) {
      let vx = m.center[0] - cx;
      let vy = m.center[1] - cy;
      if (Math.hypot(vx, vy) < 1e-3) {
        const far = m.corners.reduce((a, b) =>
          (dist(b, m.center) > dist(a, m.center) ? b : a));
        vx = far[0] - m.center[0]; vy = far[1] - m.center[1];
      }
      const norm = Math.hypot(vx, vy) || 1;
      const outward = [vx / norm, vy / norm];
      const [tip, found] = findWedgeTip(C, gray, m, outward);
      const role = STICKER_ROLE_BY_QUADRANT[
        `${outward[0] >= 0 ? 1 : -1},${outward[1] >= 0 ? 1 : -1}`
      ];
      out.push({
        marker: m, outward,
        corner_point: [tip[0], tip[1]],
        bracket_found: found,
        role: role || null,
      });
    }
    return out;
  }

  function stickerQuad(stickers) {
    if (stickers.length >= 4) return orderPoints(stickers.slice(0, 4).map((s) => s.corner_point));
    if (stickers.length === 3) {
      const byRole = {};
      for (const s of stickers) if (s.role) byRole[s.role] = s.corner_point;
      if (Object.keys(byRole).length === 3) return completeQuadFromThree(byRole);
    }
    return null;
  }

  // corner_sticker.estimate_page_size
  function estimatePageSize(stickers) {
    if (stickers.length < 3) return null;
    const quad = stickerQuad(stickers);
    if (!quad) return null;
    const scales = stickers.map((s) => markerSide(s.marker) / CORNER_STICKER_ARUCO_MM);
    if (scales.some((v) => v < 1e-6)) return null;

    let ppm;
    if (stickers.length >= 4) {
      const cps = stickers.map((s) => s.corner_point);
      ppm = quad.map((q) => {
        let bi = 0;
        let bd = Infinity;
        cps.forEach((cp, i) => { const d = dist(cp, q); if (d < bd) { bd = d; bi = i; } });
        return scales[bi];
      });
    } else {
      const mean = scales.reduce((a, b) => a + b, 0) / scales.length;
      ppm = [mean, mean, mean, mean];
    }
    const avg = (a, b) => (a + b) / 2;
    const [tl, tr, br, bl] = quad;
    const wMm = avg(dist(tr, tl) / avg(ppm[0], ppm[1]), dist(br, bl) / avg(ppm[3], ppm[2]));
    const hMm = avg(dist(bl, tl) / avg(ppm[0], ppm[3]), dist(br, tr) / avg(ppm[1], ppm[2]));

    let best = null;
    let bestErr = 1e9;
    for (const [name, [pw, ph]] of Object.entries(PAPERS_MM)) {
      for (const [a, b] of [[pw, ph], [ph, pw]]) {
        const err = Math.abs(wMm - a) + Math.abs(hMm - b);
        if (err < bestErr) { bestErr = err; best = name; }
      }
    }
    const r1 = (v) => Math.round(v * 10) / 10;
    return {
      width_mm: r1(wMm), height_mm: r1(hMm),
      px_per_mm: Math.round(ppm.reduce((a, b) => a + b, 0) / 4 * 1000) / 1000,
      method: "corner_stickers",
      best_match: bestErr < 30 ? best : null,
      match_error_mm: r1(bestErr),
    };
  }

  // literal_box.mask_literals — blank each literal interior on an RGBA image
  function maskLiterals(C, rgbaMat, assets, fill = 255) {
    for (const a of assets) {
      const [x, y, w, h] = a.bbox.map((v) => Math.round(v));
      const pad = Math.max(2, Math.floor(Math.min(w, h) * 0.12));
      C.rectangle(
        rgbaMat,
        new C.Point(x + pad, y + pad),
        new C.Point(x + w - pad, y + h - pad),
        new C.Scalar(fill, fill, fill, 255),
        -1,
      );
    }
  }

  root.WJMVision = {
    ROLE_BY_ID, ROLE_ORDER, TARGET_LONG_PX, CORNER_STICKER_ID,
    hasAruco, MarkerDetector, pageQuadFromMarkers, outputSize, rectify, tenengrad, dist,
    planTargets, anchorHomography, akazeHomography, orbHomography, landmarkMask, compositeInto,
    detectMetadataBlock, segmentLines, detectLiteralAssets, maskLiterals,
    detectRegistrationMarks, marksToQuad, laplacianVariance, assessSharpness, edgeAcutance,
    detectCornerStickers, stickerQuad, estimatePageSize,
  };
})(typeof self !== "undefined" ? self : globalThis);

if (typeof module !== "undefined" && module.exports) module.exports = globalThis.WJMVision;
