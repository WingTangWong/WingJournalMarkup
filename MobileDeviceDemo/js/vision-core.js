/* WJM vision core — runs inside the worker (classic script, no ES modules so it
 * can sit alongside importScripts('opencv.js')).
 *
 * Mirrors wingjournal.vision.{aruco,boundary,rectify} so a capture here lines up
 * with `wingjournal ingest`:
 *   dictionary  DICT_4X4_50
 *   ids         0/1/2/3 = TOP_LEFT / TOP_RIGHT / BOTTOM_RIGHT / BOTTOM_LEFT
 *   page frame  outer corner of each of the 4 markers, ordered TL,TR,BR,BL
 *   normalized  longer side 1600 px, aspect from the quad clamped to [1.15, 1.6]
 */
(function (root) {
  "use strict";

  const ROLE_BY_ID = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"];
  const ROLE_ORDER = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"];
  const TARGET_LONG_PX = 1600;
  const ASPECT_RANGE = [1.15, 1.6];

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
  function rectify(cv, srcMat, quad) {
    const [w, h] = outputSize(quad);
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, quad.flat());
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1]);
    const Hm = cv.getPerspectiveTransform(src, dst);
    const out = new cv.Mat();
    cv.warpPerspective(
      srcMat, out, Hm, new cv.Size(w, h),
      typeof cv.INTER_CUBIC === "number" ? cv.INTER_CUBIC : 1,
    );
    src.delete(); dst.delete(); Hm.delete();
    return { mat: out, width: w, height: h };
  }

  root.WJMVision = {
    ROLE_BY_ID, ROLE_ORDER, TARGET_LONG_PX,
    hasAruco, MarkerDetector, pageQuadFromMarkers, outputSize, rectify, dist,
  };
})(typeof self !== "undefined" ? self : globalThis);

if (typeof module !== "undefined" && module.exports) module.exports = self.WJMVision;
