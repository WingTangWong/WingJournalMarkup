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
 * bit0=lower-left ("None" filled = 0, all three filled = 7). Pure classical
 * CV, no dictionary, no ML model.
 *
 * Rotation/tilt tolerance: corners are read off in clockwise order starting
 * from whichever corner sits nearest the bullseye, so the UR/LR/LL labels
 * stay correct no matter how the glyph is rotated in frame — the bullseye
 * (found first, independent of the box's orientation) anchors the reading.
 *
 * v3 — built for a *high* degree of tolerance to real hand-drawn error, not
 * just the specific failure found in v1/v2. The design principle changed:
 * v1/v2 both tried to find ONE clean connected ink shape and measure it
 * precisely (a closed contour, then an open-but-still-one-blob contour).
 * That breaks the moment the ink itself is broken — a dashed/gappy line, a
 * corner that doesn't quite meet, a stroke that touches nearby handwriting
 * and gets fused into one big contour with it. v3 never depends on the box
 * being one connected shape at all:
 *
 *   - bullseye: unchanged from v2 — cv.HoughCircles (gradient voting
 *     tolerates ring gaps and stroke irregularity) confirmed by ink density
 *     in three concentric bands (dark center dot, light middle, dark ring).
 *   - box: anchored on a confirmed bullseye. Every ink pixel is grouped into
 *     its connected component once per frame (findInkComponents); a
 *     component only counts as possible box material if it's LONG relative
 *     to the bullseye (a real box edge is a couple of bullseye-diameters
 *     long; an individual handwritten letter stroke is not — this is what
 *     actually rejects nearby writing, not distance alone, since real
 *     writing is often well within any search radius generous enough to
 *     tolerate a gappy box). The fit is then SEEDED from the single
 *     qualifying component nearest the bullseye — by design the intended
 *     box sits right next to it, much closer than any neighboring glyph's
 *     box in a densely packed sheet — and grown: each round, cv.minAreaRect
 *     refits over whatever qualifying ink (from any component in range)
 *     ended up near the current fit's boundary. A gappy box's other sides
 *     (separate components) join in once the fit reaches them; a distant
 *     neighboring glyph's box never does, because seeding from "nearest
 *     first" means it's never in the running to begin with.
 *
 * Verified two ways. Synthetic (tests/glyph-detect.test.mjs, all passing):
 * every 3-bit value, an open-bracket box and a closed one, 25-degree
 * rotation, mid-line gaps, wobbly/non-parallel edges, a hatched rather than
 * solid corner mark, stray handwriting-like ink beside the glyph, two
 * glyphs at real-sheet packing density, and negative cases (blank page, a
 * plain rectangle with no bullseye).
 *
 * Real photo (a hand-drawn 4x4 reference sheet, 16 glyphs, packed tightly
 * and with label text close enough above the first glyph to have wrecked an
 * earlier version's fit): 8 of 16 located, no false positives, ~80ms at the
 * demo's live 900px processing width. The located boxes land accurately on
 * the glyphs. Decoding is the weaker half: spot-checking against the sheet
 * by eye, roughly half the values are right and the rest are still wrong.
 * v1/v2 found nothing at all on this sheet, so this is a large step, but
 * "located" is not "decoded" and neither number is finished work. The sheet
 * is also a deliberately hard case (16 glyphs at a packing density a real
 * page would rarely have).
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

  // fraction of "ink" (binary mask > 0) pixels within an annulus
  // [r0, r1] around (cx, cy) — r0 = 0 samples a solid disk. Used both to
  // confirm a Hough circle is a real bullseye (dark center / light middle /
  // dark outer ring) and, with a tiny disk, as a single-point ink probe.
  function annulusInkFrac(binary, cx, cy, r0, r1) {
    const W = binary.cols;
    const H = binary.rows;
    const data = binary.data;
    const x0 = Math.max(0, Math.floor(cx - r1));
    const x1 = Math.min(W - 1, Math.ceil(cx + r1));
    const y0 = Math.max(0, Math.floor(cy - r1));
    const y1 = Math.min(H - 1, Math.ceil(cy + r1));
    const r0sq = r0 * r0;
    const r1sq = r1 * r1;
    let hit = 0;
    let total = 0;
    for (let y = y0; y <= y1; y++) {
      const rowOff = y * W;
      const dy = y - cy;
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const d2 = dx * dx + dy * dy;
        if (d2 < r0sq || d2 > r1sq) continue;
        total++;
        if (data[rowOff + x] > 0) hit++;
      }
    }
    return total ? hit / total : 0;
  }

  // cv.HoughCircles finds circular edge structure directly from gradients —
  // tolerant of a gap in the ring or an uneven/retraced stroke, unlike
  // requiring a topologically closed contour. Confirmed against the ink mask:
  // dark center (the dot), light middle band, dark outer ring.
  function findBullseyes(C, gray, binary, lo, hi) {
    const minR = Math.max(4, lo / 2);
    const maxR = Math.max(minR + 1, hi / 2);
    const minDist = Math.max(20, lo);

    const med = new C.Mat();
    C.medianBlur(gray, med, 5);
    const circles = new C.Mat();
    C.HoughCircles(med, circles, C.HOUGH_GRADIENT, 1.2, minDist, 80, 28, minR, maxR);
    med.delete();

    const found = [];
    const n = circles.data32F.length / 3;
    for (let i = 0; i < n; i++) {
      const cx = circles.data32F[i * 3];
      const cy = circles.data32F[i * 3 + 1];
      const r = circles.data32F[i * 3 + 2];
      const centerInk = annulusInkFrac(binary, cx, cy, 0, r * 0.25);
      const midInk = annulusInkFrac(binary, cx, cy, r * 0.35, r * 0.68);
      // wider than the ring's nominal footprint on purpose: Hough's own
      // radius estimate can be off by a fair bit under rotation/interpolation
      // or a wobbly hand-drawn ring, so a tight band can miss ink that's
      // really there
      const ringInk = annulusInkFrac(binary, cx, cy, r * 0.7, r * 1.2);
      if (centerInk > 0.4 && midInk < 0.42 && ringInk > 0.28) {
        found.push({ cx, cy, diameter: r * 2 });
      }
    }
    circles.delete();
    return found;
  }

  function rotatedRectCorners(rect) {
    const { center, size, angle } = rect;
    const a = (angle * Math.PI) / 180;
    const cos = Math.cos(a);
    const sin = Math.sin(a);
    const hw = size.width / 2;
    const hh = size.height / 2;
    return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([lx, ly]) => ({
      x: center.x + lx * cos - ly * sin,
      y: center.y + lx * sin + ly * cos,
    }));
  }

  function hasInkNear(binary, x, y, half) {
    const W = binary.cols;
    const H = binary.rows;
    const data = binary.data;
    const xi = Math.round(x);
    const yi = Math.round(y);
    half = Math.max(1, Math.round(half)); // must be integer: used as a raw pixel-index offset below
    for (let dy = -half; dy <= half; dy++) {
      const yy = yi + dy;
      if (yy < 0 || yy >= H) continue;
      const rowOff = yy * W;
      for (let dx = -half; dx <= half; dx++) {
        const xx = xi + dx;
        if (xx < 0 || xx >= W) continue;
        if (data[rowOff + xx] > 0) return true;
      }
    }
    return false;
  }

  // fraction of sample points along a straight edge that have ink nearby —
  // tolerant of a wavy or dashed hand-drawn line, unlike requiring the exact
  // edge pixel or unbroken ink.
  function edgeInkFrac(binary, p0, p1, samples, halfWidth) {
    let hit = 0;
    for (let i = 0; i < samples; i++) {
      const t = i / (samples - 1);
      const x = p0.x + (p1.x - p0.x) * t;
      const y = p0.y + (p1.y - p0.y) * t;
      if (hasInkNear(binary, x, y, halfWidth)) hit++;
    }
    return hit / samples;
  }

  // Every ink pixel belongs to some connected component (a contour). Split
  // the whole frame into components once per frame — cheap, and lets the box
  // fit below filter by component LENGTH, not just proximity: a real box
  // edge is long (a couple of bullseye diameters), while an individual
  // handwritten letter stroke is short. That's what actually keeps nearby
  // writing (a title above the glyph, another glyph's label) out of the fit
  // — proximity alone doesn't, real writing is often well within any search
  // radius generous enough to tolerate a gappy box.
  function findInkComponents(C, binary) {
    const contours = new C.MatVector();
    const hierarchy = new C.Mat();
    C.findContours(binary, contours, hierarchy, C.RETR_LIST, C.CHAIN_APPROX_NONE);
    hierarchy.delete();
    const comps = [];
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const r = C.boundingRect(cnt);
      const pts = Array.from(cnt.data32S);
      cnt.delete();
      comps.push({
        longSide: Math.max(r.width, r.height),
        cx: r.x + r.width / 2,
        cy: r.y + r.height / 2,
        radius: Math.hypot(r.width, r.height) / 2,
        pts,
      });
    }
    contours.delete();
    return comps;
  }

  // every ink pixel within radius r of (cx, cy), from components at least
  // minLen long, excluding a disk around the center (so the bullseye's own
  // ring/dot ink isn't counted as box ink). Flat [x0,y0,x1,y1,...] array, not
  // {x,y} objects — this can be a lot of points and avoiding per-point
  // object allocation matters here.
  function collectLineInkPoints(components, cx, cy, r, excludeR, minLen) {
    const rsq = r * r;
    const exsq = excludeR * excludeR;
    const pts = [];
    for (const comp of components) {
      if (comp.longSide < minLen) continue;
      const dx0 = comp.cx - cx;
      const dy0 = comp.cy - cy;
      const reach = r + comp.radius;
      if (dx0 * dx0 + dy0 * dy0 > reach * reach) continue; // component nowhere near the search disk
      for (let i = 0; i < comp.pts.length; i += 2) {
        const x = comp.pts[i];
        const y = comp.pts[i + 1];
        const dx = x - cx;
        const dy = y - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 > rsq || d2 < exsq) continue;
        pts.push(x, y);
      }
    }
    return pts;
  }

  function minAreaRectFromFlatPoints(C, flat) {
    const n = flat.length / 2;
    if (n < 5) return null;
    const mat = C.matFromArray(n, 1, C.CV_32SC2, flat);
    const rect = C.minAreaRect(mat);
    mat.delete();
    return rect;
  }

  function pointToSegmentDist(px, py, ax, ay, bx, by) {
    const abx = bx - ax;
    const aby = by - ay;
    const apx = px - ax;
    const apy = py - ay;
    const abLen2 = abx * abx + aby * aby;
    let t = abLen2 > 0 ? (apx * abx + apy * aby) / abLen2 : 0;
    t = Math.max(0, Math.min(1, t));
    const qx = ax + abx * t;
    const qy = ay + aby * t;
    return Math.hypot(px - qx, py - qy);
  }

  // ink fraction in a disk, counting ONLY pixels far enough from the fitted
  // box's own outline to not be part of it — see bitAt's note on why a plain
  // disk probe can't tell a filled corner from an empty one.
  function cornerInkFrac(binary, rectCorners, cx, cy, radius, borderTol) {
    const W = binary.cols;
    const H = binary.rows;
    const data = binary.data;
    const x0 = Math.max(0, Math.floor(cx - radius));
    const x1 = Math.min(W - 1, Math.ceil(cx + radius));
    const y0 = Math.max(0, Math.floor(cy - radius));
    const y1 = Math.min(H - 1, Math.ceil(cy + radius));
    const rsq = radius * radius;
    let hit = 0;
    let total = 0;
    for (let y = y0; y <= y1; y++) {
      const rowOff = y * W;
      const dy = y - cy;
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        if (dx * dx + dy * dy > rsq) continue;
        if (distToRectBoundary(rectCorners, x, y) < borderTol) continue;
        total++;
        if (data[rowOff + x] > 0) hit++;
      }
    }
    return total ? hit / total : 0;
  }

  function distToRectBoundary(corners, x, y) {
    let best = Infinity;
    for (let k = 0; k < 4; k++) {
      const a = corners[k];
      const b = corners[(k + 1) % 4];
      const d = pointToSegmentDist(x, y, a.x, a.y, b.x, b.y);
      if (d < best) best = d;
    }
    return best;
  }

  // The core of v3's tolerance: don't require the box to be one connected
  // shape. Gather ink pixels in a generous radius around the bullseye — but
  // only from components long enough to plausibly be a box edge, not a
  // handwritten letter stroke — fit a rotated rectangle, then iteratively
  // keep only the points that ended up near that fit's boundary and refit —
  // a few rounds of this converges onto the box even when its lines are
  // gappy, uneven, non-parallel, or when unrelated ink (nearby handwriting,
  // another glyph) sits within the same search radius.
  function fitBoxForBullseye(C, binary, components, bullseye, lo) {
    const { cx, cy, diameter } = bullseye;
    const W = binary.cols;
    const H = binary.rows;
    // measured against a real sheet: a box's far corner sits roughly 3-3.2x
    // the bullseye's diameter away, while glyphs in a densely packed grid
    // can sit only ~5-6x apart — this needs enough margin over the former
    // without reaching the latter
    const searchR = Math.min(diameter * 4.5, Math.max(W, H));
    // generous on purpose: Hough's own radius estimate can undershoot the
    // ring's true outer edge (seen under rotation/interpolation — the ring's
    // real ink can reach past 1x the reported diameter/2), and any ring
    // pixel that leaks past this boundary badly skews the box fit toward
    // including the bullseye itself
    const excludeR = diameter * 1.05;
    // a real box's shorter side still comfortably clears this — measured
    // against a real photo, a stray title-text stroke sitting well inside
    // the search radius did not
    const minLen = diameter * 0.9;

    const pool = collectLineInkPoints(components, cx, cy, searchR, excludeR, minLen);
    if (pool.length / 2 < 24) return null;

    // seed the fit from the SINGLE qualifying component nearest the
    // bullseye, not the whole pool at once. By design the intended box sits
    // immediately next to the bullseye (a small gap, not several diameters)
    // — much closer than any neighboring glyph's box in a densely packed
    // grid — so this reliably grabs the right one even when the full search
    // radius (needed for gap tolerance) also reaches other glyphs' boxes.
    let seed = null;
    let seedDist = Infinity;
    const exsq = excludeR * excludeR;
    for (const comp of components) {
      if (comp.longSide < minLen) continue;
      for (let i = 0; i < comp.pts.length; i += 2) {
        const x = comp.pts[i];
        const y = comp.pts[i + 1];
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (d2 < exsq || d2 > searchR * searchR) continue;
        if (d2 < seedDist) { seedDist = d2; seed = comp; }
      }
    }
    if (!seed) return null;

    let rect = minAreaRectFromFlatPoints(C, seed.pts);
    if (!rect) return null;

    // grow from the seed: each round, pull in ink from the whole pool (any
    // qualifying component in range, not just the seed's) that ended up
    // near the CURRENT fit's boundary, then refit. A gappy box's other
    // sides (separate components) join in once the fit reaches them; a
    // distant neighboring glyph's box never does, because it isn't near
    // this fit's boundary no matter how many rounds run.
    // enough rounds for a fit seeded from a small fragment (a gap cut off
    // most of the box) to fully grow out to the true shape — convergence is
    // gradual (one edge's worth of expansion per round is typical), and
    // stopping early leaves an undersized, wrong-value rect that still
    // passes the sanity gates below
    for (let iter = 0; iter < 10; iter++) {
      const corners = rotatedRectCorners(rect);
      // floor anchored to the bullseye's diameter, not just the current
      // fit's own size: when the seed is itself a small fragment (a gap cut
      // it short), its minAreaRect is tiny/thin, and a tolerance scaled
      // only off THAT never reaches far enough to pull in the box's other,
      // separately-gapped sides — the fit would get stuck at the seed's own
      // shape forever
      const tol = Math.max(16, diameter * 0.5, 0.07 * Math.max(rect.size.width, rect.size.height));
      const kept = [];
      for (let i = 0; i < pool.length; i += 2) {
        const x = pool[i];
        const y = pool[i + 1];
        if (distToRectBoundary(corners, x, y) < tol) { kept.push(x, y); }
      }
      if (kept.length / 2 < 20) break;
      const next = minAreaRectFromFlatPoints(C, kept);
      if (!next) break;
      rect = next;
    }

    const shortSide = Math.min(rect.size.width, rect.size.height);
    const longSide = Math.max(rect.size.width, rect.size.height);
    // A real glyph's box is comfortably bigger than its bullseye (measured
    // ~2.3-2.6x on a real sheet). Requiring a clear margin over the bullseye
    // in BOTH dimensions is what stops a round letter bowl plus a
    // text-sized scrap of ink from passing as a glyph — that exact false
    // positive showed up on the handwritten title of a real photo. Still
    // generous, not a precise ratio: proportions aren't tightly specified.
    if (shortSide < Math.max(lo * 0.4, diameter * 1.2)) return null;
    if (longSide < diameter * 1.6 || longSide > diameter * 9) return null;

    const corners = rotatedRectCorners(rect);
    // the bullseye should sit right at (or just outside) the fitted box's
    // nearest corner — this is what keeps a search radius generous enough
    // for gappy ink from also drifting onto a *different*, closer glyph's
    // box in a densely packed sheet
    const nearest = Math.min(...corners.map((p) => Math.hypot(p.x - cx, p.y - cy)));
    if (nearest > diameter * 1.6) return null;

    let sidesOk = 0;
    for (let k = 0; k < 4; k++) {
      const halfWidth = Math.max(5, Math.round(shortSide * 0.045));
      if (edgeInkFrac(binary, corners[k], corners[(k + 1) % 4], 30, halfWidth) > 0.18) sidesOk++;
    }
    // 2 of 4, not 3: the convention already omits one side outright, and a
    // heavily gapped real side can legitimately read low here even when the
    // overall fit (seeded from a real, length-filtered component, checked
    // against the bullseye's own position above) is correct
    if (sidesOk < 2) return null;

    return corners;
  }

  /** @param {cv.Mat} gray single-channel */
  function detectGlyphs(C, gray) {
    const H = gray.rows;
    const W = gray.cols;
    const maxDim = Math.max(H, W);
    const lo = 0.012 * maxDim; // min bullseye diameter
    // max bullseye diameter — HoughCircles' cost scales heavily with the
    // radius range searched (a 0.16 upper bound measured ~2.7s on a
    // 1320-wide real photo; 0.08 measured ~150-250ms with no loss of the
    // real bullseye, which sits well inside it), so keep this tight
    const hi = 0.08 * maxDim;

    const blur = new C.Mat();
    const binary = new C.Mat();
    C.GaussianBlur(gray, blur, new C.Size(3, 3), 0);
    C.threshold(blur, binary, 0, 255, C.THRESH_BINARY_INV | C.THRESH_OTSU);
    blur.delete();

    const bullseyes = findBullseyes(C, gray, binary, lo, hi);
    const components = findInkComponents(C, binary);

    const glyphs = [];
    for (const b of bullseyes) {
      const corners = fitBoxForBullseye(C, binary, components, b, lo);
      if (!corners) continue;

      const bPt = { x: b.cx, y: b.cy };
      const cw = sortClockwise(corners);
      let anchorIndex = 0;
      let anchorD = Infinity;
      cw.forEach((p, i) => {
        const d = Math.hypot(p.x - bPt.x, p.y - bPt.y);
        if (d < anchorD) { anchorD = d; anchorIndex = i; }
      });
      const ur = cw[(anchorIndex + 1) % 4];
      const lr = cw[(anchorIndex + 2) % 4];
      const ll = cw[(anchorIndex + 3) % 4];
      const c = centroid(cw);

      const bitAt = (corner) => {
        const dx = c.x - corner.x;
        const dy = c.y - corner.y;
        const dd = Math.hypot(dx, dy) || 1;
        const inset = 0.25 * dd;
        const px = corner.x + (dx / dd) * inset;
        const py = corner.y + (dy / dd) * inset;
        const radius = Math.max(5, Math.round(0.17 * dd));
        // Ignore ink belonging to the box's own outline. Two border strokes
        // meet at every corner, so a plain disk probe here reads "inked" on
        // an EMPTY corner just as readily as a filled one — measured against
        // a real sheet, that alone turned 1s and 4s into 5s and 7s. The
        // threshold stays low so a hatched or scribbled mark still counts.
        const borderTol = Math.max(6, 0.08 * dd);
        return cornerInkFrac(binary, cw, px, py, radius, borderTol) > 0.22 ? 1 : 0;
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
