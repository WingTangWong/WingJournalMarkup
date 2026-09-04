/* Bullseye + 3-bit corner-code glyph detector (classic script, runs inside the
 * worker alongside importScripts('opencv.js') — no ES modules).
 *
 * This is a *different* fiducial from WJM's ArUco corner markers. It's a
 * hand-drawable glyph (see the "Proposed Handwritten Syntax Lexicon" sketch):
 *
 *   - a BULLSEYE (a ring with a dot in its center) sits just outside one
 *     corner of a box outline, with a small gap between them — this is the
 *     glyph's fixed anchor, always present, and by convention marks the
 *     box's "upper-left" corner.
 *   - the box's other three corners (upper-right, lower-right, lower-left,
 *     walking clockwise from the anchor) may each carry a small filled
 *     square patch just inside that corner, or be left plain.
 *
 * Those three corners are a 3-bit code: bit2=upper-right, bit1=lower-right,
 * bit0=lower-left ("None" filled = 0, all three filled = 7). Detection is
 * pure classical CV — nested-contour shape analysis for the bullseye (the
 * same technique src/wingjournal/vision/registration.py and this demo's
 * sibling MobileDeviceDemo/js/vision-core.js use for the concentric-square
 * registration marks, just circular instead of square), approxPolyDP for
 * the box outline, and a small ink-density probe at each candidate corner.
 * No ArUco dictionary, no ML model.
 *
 * Rotation/tilt tolerance: corners are read off in clockwise order starting
 * from whichever corner sits nearest the bullseye, so the UR/LR/LL labels
 * stay correct no matter how the glyph is rotated in frame — the bullseye
 * (found first, independent of the box's orientation) anchors the reading.
 *
 * Tuning knobs below (circularity/size/density thresholds) are first-pass
 * numbers, not calibrated against real hand-drawn samples yet — expect to
 * revisit them once this runs against real photos, the same way the
 * project's OCR/HTR tuning has gone.
 *
 * detectGlyphs(cv, gray) -> [{
 *   quad: [[x,y]x4]  clockwise, quad[anchorIndex] is the corner nearest the bullseye
 *   anchorIndex: 0..3
 *   bullseye: {cx, cy, diameter}
 *   bits: {upperRight, lowerRight, lowerLeft}  each 0|1
 *   value: 0..7
 * }]
 */
(function (root) {
  "use strict";

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  function centroid(pts) {
    let x = 0;
    let y = 0;
    for (const p of pts) { x += p.x; y += p.y; }
    return { x: x / pts.length, y: y / pts.length };
  }

  // clockwise order on screen (y grows downward, so ascending atan2 sweeps
  // right -> down -> left -> up, i.e. clockwise)
  function sortClockwise(pts) {
    const c = centroid(pts);
    return pts.slice().sort(
      (a, b) => Math.atan2(a.y - c.y, a.x - c.x) - Math.atan2(b.y - c.y, b.x - c.x),
    );
  }

  // roughly-circular blob test — tolerant of hand-drawn wobble on purpose.
  // Works the same whether the contour is a filled dot or a ring's outer
  // boundary (contourArea/arcLength only see the outer edge either way).
  function circleish(C, cnt) {
    const area = C.contourArea(cnt);
    if (area < 8) return null;
    const r = C.boundingRect(cnt);
    if (Math.min(r.width, r.height) === 0) return null;
    if (Math.abs(r.width - r.height) > 0.45 * Math.max(r.width, r.height)) return null;
    const peri = C.arcLength(cnt, true);
    if (peri <= 0) return null;
    const circularity = (4 * Math.PI * area) / (peri * peri); // 1.0 = perfect circle
    if (circularity < 0.5) return null;
    return { cx: r.x + r.width / 2, cy: r.y + r.height / 2, diameter: Math.max(r.width, r.height) };
  }

  function quadFromContour(C, cnt) {
    const area = C.contourArea(cnt);
    if (area < 60) return null;
    const peri = C.arcLength(cnt, true);
    const approx = new C.Mat();
    C.approxPolyDP(cnt, approx, 0.02 * peri, true);
    let corners = null;
    if (approx.rows === 4 && C.isContourConvex(approx)) {
      corners = [];
      for (let i = 0; i < 4; i++) {
        corners.push({ x: approx.data32S[i * 2], y: approx.data32S[i * 2 + 1] });
      }
    }
    approx.delete();
    if (!corners) return null;
    const r = C.boundingRect(cnt);
    if (area / (r.width * r.height) < 0.55) return null; // reject very non-rectangular quads
    return corners;
  }

  // hierarchy[i*4 + {0:next,1:prev,2:child,3:parent}] — RETR_TREE layout
  function findBullseyes(C, contours, hd, lo, hi) {
    const found = [];
    for (let i = 0; i < contours.size(); i++) {
      if (hd[i * 4 + 3] !== -1) continue; // top-level only: the ring's outer edge
      const cnt = contours.get(i);
      const outer = circleish(C, cnt);
      cnt.delete();
      if (!outer || outer.diameter < lo || outer.diameter > hi) continue;

      const holeIdx = hd[i * 4 + 2];
      if (holeIdx === -1) continue; // no hole -> not a ring, just a filled blob
      const holeCnt = contours.get(holeIdx);
      const hole = circleish(C, holeCnt);
      holeCnt.delete();
      if (!hole || hole.diameter < 0.3 * outer.diameter || hole.diameter > 0.92 * outer.diameter) continue;

      // the center dot sits as a child of the hole (a foreground blob inside
      // the ring's background-colored interior) — walk the hole's children
      let dot = null;
      let d = hd[holeIdx * 4 + 2];
      while (d !== -1) {
        const dotCnt = contours.get(d);
        const cand = circleish(C, dotCnt);
        dotCnt.delete();
        if (cand && cand.diameter >= 0.06 * outer.diameter && cand.diameter <= 0.65 * outer.diameter
          && Math.hypot(cand.cx - outer.cx, cand.cy - outer.cy) < 0.4 * outer.diameter) {
          dot = cand;
          break;
        }
        d = hd[d * 4]; // next sibling
      }
      if (!dot) continue;

      found.push({ cx: outer.cx, cy: outer.cy, diameter: outer.diameter });
    }
    return found;
  }

  function longSide(q) {
    const xs = q.map((p) => p.x);
    const ys = q.map((p) => p.y);
    return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  }

  /** @param {cv.Mat} gray single-channel */
  function detectGlyphs(C, gray) {
    const H = gray.rows;
    const W = gray.cols;
    const maxDim = Math.max(H, W);
    const lo = 0.012 * maxDim;
    const hi = 0.16 * maxDim;

    const blur = new C.Mat();
    const binary = new C.Mat();
    C.GaussianBlur(gray, blur, new C.Size(3, 3), 0);
    C.threshold(blur, binary, 0, 255, C.THRESH_BINARY_INV | C.THRESH_OTSU);
    blur.delete();

    const contours = new C.MatVector();
    const hierarchy = new C.Mat();
    C.findContours(binary, contours, hierarchy, C.RETR_TREE, C.CHAIN_APPROX_SIMPLE);
    const hd = hierarchy.data32S;

    const bullseyes = findBullseyes(C, contours, hd, lo, hi);

    const quads = [];
    for (let i = 0; i < contours.size(); i++) {
      if (hd[i * 4 + 3] !== -1) continue;
      const cnt = contours.get(i);
      const q = quadFromContour(C, cnt);
      cnt.delete();
      if (q) quads.push(q);
    }
    contours.delete();
    hierarchy.delete();

    // pair each bullseye with its nearest unclaimed, plausibly-sized quad —
    // closest matches win first so two glyphs near each other don't steal
    // one another's box
    const claimed = new Set();
    const ranked = bullseyes.map((b) => {
      const bPt = { x: b.cx, y: b.cy };
      let best = -1;
      let bestD = Infinity;
      quads.forEach((q, qi) => {
        const long = longSide(q);
        if (long < 1.1 * b.diameter || long > 12 * b.diameter) return;
        const d = Math.min(...q.map((p) => dist(p, bPt)));
        if (d < bestD) { bestD = d; best = qi; }
      });
      return { b, bPt, best, bestD };
    }).sort((x, y) => x.bestD - y.bestD);

    const glyphs = [];
    for (const { b, bPt, best, bestD } of ranked) {
      if (best === -1 || !isFinite(bestD) || bestD > 5 * b.diameter || claimed.has(best)) continue;
      claimed.add(best);

      const cw = sortClockwise(quads[best]);
      let anchorIndex = 0;
      let anchorD = Infinity;
      cw.forEach((p, i) => {
        const d = dist(p, bPt);
        if (d < anchorD) { anchorD = d; anchorIndex = i; }
      });
      const ur = cw[(anchorIndex + 1) % 4];
      const lr = cw[(anchorIndex + 2) % 4];
      const ll = cw[(anchorIndex + 3) % 4];
      const c = centroid(cw);
      const avgSide = (dist(cw[0], cw[1]) + dist(cw[1], cw[2]) + dist(cw[2], cw[3]) + dist(cw[3], cw[0])) / 4;

      const bitAt = (corner) => {
        const dx = c.x - corner.x;
        const dy = c.y - corner.y;
        const dd = Math.hypot(dx, dy) || 1;
        const inset = 0.3 * dd;
        const px = corner.x + (dx / dd) * inset;
        const py = corner.y + (dy / dd) * inset;
        const half = Math.max(3, Math.round(0.16 * avgSide));
        const x0 = Math.max(0, Math.round(px - half));
        const y0 = Math.max(0, Math.round(py - half));
        const x1 = Math.min(W, Math.round(px + half));
        const y1 = Math.min(H, Math.round(py + half));
        if (x1 <= x0 || y1 <= y0) return 0;
        const roi = binary.roi(new C.Rect(x0, y0, x1 - x0, y1 - y0));
        const inkFrac = C.mean(roi)[0] / 255;
        roi.delete();
        return inkFrac > 0.38 ? 1 : 0;
      };

      const bits = { upperRight: bitAt(ur), lowerRight: bitAt(lr), lowerLeft: bitAt(ll) };
      const value = (bits.upperRight << 2) | (bits.lowerRight << 1) | bits.lowerLeft;

      glyphs.push({
        quad: cw.map((p) => [p.x, p.y]),
        anchorIndex,
        bullseye: { cx: b.cx, cy: b.cy, diameter: b.diameter },
        bits,
        value,
      });
    }

    binary.delete();
    return glyphs;
  }

  root.WJMGlyphs = { detectGlyphs };
})(typeof self !== "undefined" ? self : globalThis);

if (typeof module !== "undefined" && module.exports) module.exports = globalThis.WJMGlyphs;
