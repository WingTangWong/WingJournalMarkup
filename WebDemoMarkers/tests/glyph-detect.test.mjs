/* Smoke test for js/glyph-detect.js against the vendored OpenCV.js (WASM).
 * Draws synthetic bullseye+box+3-bit-code glyphs at realistic scale, in the
 * real convention diagnosed against an actual photo (the box drawn as an
 * open 3-sided bracket — the side facing the bullseye is left out; see
 * WebDemoMarkers/README.md "Status" and js/glyph-detect.js's header for how
 * that was found and why the detector is built the way it is).
 *
 *   node tests/glyph-detect.test.mjs        (needs vendor/opencv.js)
 */
import { createRequire } from "module";
import assert from "assert";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const cv = require(path.join(root, "vendor/opencv.js"));

let pass = 0;
const t = (name, fn) => {
  try { fn(); pass++; console.log("ok  " + name); }
  catch (e) { console.error("FAIL " + name + "\n     " + (e.stack || e)); process.exitCode = 1; }
};

cv.onRuntimeInitialized = () => {
  globalThis.cv = cv;
  require(path.join(root, "js/glyph-detect.js"));
  const G = globalThis.WJMGlyphs;

  // Draws one glyph at roughly the scale seen in a real phone photo of a
  // hand-drawn sheet: a bullseye (ring+dot) just outside the box's top-left
  // corner, with 0/1 fills at the other three corners.
  //
  // open=true (the real convention): the box's left side (facing the
  // bullseye) is left undrawn — the detector must recover the rectangle
  // from minAreaRect over the remaining 3 sides, not a closed 4-gon.
  function drawGlyph(mat, bx, by, bw, bh, bits, { open = true } = {}) {
    const black = new cv.Scalar(0);
    const th = 5;
    cv.line(mat, new cv.Point(bx, by), new cv.Point(bx + bw, by), black, th);           // top
    cv.line(mat, new cv.Point(bx + bw, by), new cv.Point(bx + bw, by + bh), black, th); // right
    cv.line(mat, new cv.Point(bx + bw, by + bh), new cv.Point(bx, by + bh), black, th); // bottom
    if (!open) cv.line(mat, new cv.Point(bx, by + bh), new cv.Point(bx, by), black, th); // left

    // bullseye: a thin ring outline (a real pen stroke, not a thick filled
    // annulus — a filled-disk-minus-filled-disk ring is thick enough to give
    // Hough two comparably strong concentric edges, and it isn't what a
    // single pen stroke actually looks like) + a separate center dot
    const cx = bx - 55, cy = by - 55;
    cv.circle(mat, new cv.Point(cx, cy), 30, black, 6);
    cv.circle(mat, new cv.Point(cx, cy), 8, black, -1);

    const S = 55; // corner fill square size
    if (bits.upperRight) cv.rectangle(mat, new cv.Point(bx + bw - S, by), new cv.Point(bx + bw, by + S), black, -1);
    if (bits.lowerRight) cv.rectangle(mat, new cv.Point(bx + bw - S, by + bh - S), new cv.Point(bx + bw, by + bh), black, -1);
    if (bits.lowerLeft) cv.rectangle(mat, new cv.Point(bx, by + bh - S), new cv.Point(bx + S, by + bh), black, -1);
  }

  // box:bullseye proportions matched to the real photo this detector was
  // validated against (box longer side ~2.3-2.6x the bullseye's diameter)
  const W = 1400, H = 1700, BW = 175, BH = 205;
  const cases = [
    { value: 0, bits: { upperRight: 0, lowerRight: 0, lowerLeft: 0 } },
    { value: 5, bits: { upperRight: 1, lowerRight: 0, lowerLeft: 1 } },
    { value: 7, bits: { upperRight: 1, lowerRight: 1, lowerLeft: 1 } },
    { value: 2, bits: { upperRight: 0, lowerRight: 1, lowerLeft: 0 } },
  ];

  for (const { value, bits } of cases) {
    t(`decodes value ${value} (0b${value.toString(2).padStart(3, "0")}), open bracket box`, () => {
      const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
      drawGlyph(sheet, 400, 450, BW, BH, bits);
      const glyphs = G.detectGlyphs(cv, sheet);
      sheet.delete();
      assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
      assert.equal(glyphs[0].value, value);
      assert.deepEqual(glyphs[0].bits, bits);
    });
  }

  t("also decodes a fully-closed box (baseline, not just the open convention)", () => {
    const bits = { upperRight: 1, lowerRight: 0, lowerLeft: 0 }; // value 4
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 400, 450, BW, BH, bits, { open: false });
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
    assert.equal(glyphs[0].value, 4);
  });

  // This one failed for a long stretch of development (rotation + two solid
  // corner-fill blocks merged into the box's outline component threw the
  // fit off). What actually fixed it was not a rotation-specific tweak but
  // the growth loop's tolerance floor being anchored to the bullseye's
  // diameter instead of the current (possibly tiny) fit's own size — worth
  // keeping in mind if it ever regresses.
  t("survives a 25deg rotation (clockwise UR/LR/LL walk stays correct)", () => {
    const bits = { upperRight: 1, lowerRight: 0, lowerLeft: 1 }; // value 5
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 400, 450, BW, BH, bits);

    const center = new cv.Point(400 + BW / 2, 450 + BH / 2);
    const M = cv.getRotationMatrix2D(center, 25, 1.0);
    const rotated = new cv.Mat();
    cv.warpAffine(sheet, rotated, M, new cv.Size(W, H), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255));
    sheet.delete();
    M.delete();

    const glyphs = G.detectGlyphs(cv, rotated);
    rotated.delete();
    assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
    assert.equal(glyphs[0].value, 5);
  });

  t("two glyphs in one frame both decode without cross-pairing", () => {
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 400, 300, BW, BH, { upperRight: 1, lowerRight: 1, lowerLeft: 0 }); // 6
    drawGlyph(sheet, 400, 950, BW, BH, { upperRight: 0, lowerRight: 0, lowerLeft: 1 }); // 1
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 2, `expected two glyphs, got ${glyphs.length}`);
    const values = glyphs.map((g) => g.value).sort();
    assert.deepEqual(values, [1, 6]);
  });

  // ---- adversarial tolerance cases (the actual ask: gaps, irregular width,
  // non-parallel/wobbly lines, soft corners, partial fills, stray writing
  // crossing into the space) — each drawn deliberately messier than the
  // cases above, not just re-testing the same clean geometry.

  t("tolerates gaps mid-line, not just the missing bracket side", () => {
    const black = new cv.Scalar(0);
    const th = 5;
    const bx = 400, by = 450, bw = BW, bh = BH, gap = 18;
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    const topMid = bx + bw / 2;
    cv.line(sheet, new cv.Point(bx, by), new cv.Point(topMid - gap / 2, by), black, th);
    cv.line(sheet, new cv.Point(topMid + gap / 2, by), new cv.Point(bx + bw, by), black, th);
    const rightMid = by + bh / 2;
    cv.line(sheet, new cv.Point(bx + bw, by), new cv.Point(bx + bw, rightMid - gap / 2), black, th);
    cv.line(sheet, new cv.Point(bx + bw, rightMid + gap / 2), new cv.Point(bx + bw, by + bh), black, th);
    const botMid = bx + bw / 2;
    cv.line(sheet, new cv.Point(bx + bw, by + bh), new cv.Point(botMid + gap / 2, by + bh), black, th);
    cv.line(sheet, new cv.Point(botMid - gap / 2, by + bh), new cv.Point(bx, by + bh), black, th);
    const cx = bx - 55, cy = by - 55;
    cv.circle(sheet, new cv.Point(cx, cy), 30, black, 6);
    cv.circle(sheet, new cv.Point(cx, cy), 8, black, -1);
    const S = 55;
    cv.rectangle(sheet, new cv.Point(bx, by + bh - S), new cv.Point(bx + S, by + bh), black, -1); // lowerLeft=1

    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
    assert.equal(glyphs[0].value, 1);
  });

  t("tolerates wobbly, non-parallel (not axis-aligned) lines", () => {
    const black = new cv.Scalar(0);
    const th = 5;
    const bx = 400, by = 450, bw = BW, bh = BH;
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    // each "straight" edge drawn as a short jittered polyline instead of one
    // clean cv.line — corners stay anchored, the line between them doesn't
    const wobbly = (x0, y0, x1, y1) => {
      const n = 4;
      let px = x0, py = y0;
      for (let i = 1; i <= n; i++) {
        const t = i / n;
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        if (i < n) { x += Math.sin(i * 2.3) * 9; y += Math.cos(i * 1.9) * 9; }
        cv.line(sheet, new cv.Point(px, py), new cv.Point(x, y), black, th);
        px = x; py = y;
      }
    };
    wobbly(bx, by, bx + bw, by);
    wobbly(bx + bw, by, bx + bw, by + bh);
    wobbly(bx + bw, by + bh, bx, by + bh);
    const cx = bx - 55, cy = by - 55;
    cv.circle(sheet, new cv.Point(cx, cy), 30, black, 6);
    cv.circle(sheet, new cv.Point(cx, cy), 8, black, -1);
    const S = 55;
    cv.rectangle(sheet, new cv.Point(bx + bw - S, by), new cv.Point(bx + bw, by + S), black, -1); // upperRight=1

    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
    assert.equal(glyphs[0].value, 4);
  });

  t("tolerates a hatched (not solid-filled) corner mark", () => {
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 400, 450, BW, BH, { upperRight: 0, lowerRight: 0, lowerLeft: 0 });
    // hand-hatch the lowerLeft corner instead of using drawGlyph's solid fill
    const black = new cv.Scalar(0);
    const bx = 400, by = 450, bh = BH, S = 55;
    const x0 = bx, y0 = by + bh - S;
    for (let off = 6; off < S; off += 9) {
      cv.line(sheet, new cv.Point(x0, y0 + off), new cv.Point(x0 + off, y0), black, 3);
    }
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
    assert.equal(glyphs[0].bits.lowerLeft, 1, "hatched corner should still read as filled");
  });

  t("ignores stray handwriting-like ink near (but not on) the glyph", () => {
    const black = new cv.Scalar(0);
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 400, 450, BW, BH, { upperRight: 1, lowerRight: 0, lowerLeft: 0 }); // 4
    // short unrelated strokes above the glyph — like a label or a stray
    // word, well within the search radius but not touching the box
    for (const [x0, y0, x1, y1] of [
      [420, 380, 445, 395], [455, 375, 470, 400], [480, 385, 500, 378],
      [520, 390, 535, 405], [545, 380, 560, 392], [430, 340, 460, 355],
    ]) cv.line(sheet, new cv.Point(x0, y0), new cv.Point(x1, y1), black, 4);

    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 1, `expected exactly one glyph (not one per stray mark), got ${glyphs.length}`);
    assert.equal(glyphs[0].value, 4);
  });

  t("two glyphs packed at real-sheet density both decode without contamination", () => {
    // spacing measured off the real photo this detector was validated
    // against — tight enough that an earlier version's search radius
    // bled from one glyph's box into the next
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 400, 450, BW, BH, { upperRight: 1, lowerRight: 1, lowerLeft: 1 }); // 7
    drawGlyph(sheet, 750, 450, BW, BH, { upperRight: 0, lowerRight: 1, lowerLeft: 0 }); // 2
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 2, `expected two glyphs, got ${glyphs.length}`);
    const values = glyphs.map((g) => g.value).sort();
    assert.deepEqual(values, [2, 7]);
  });

  t("a blank page produces no glyphs", () => {
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 0);
  });

  t("a plain unrelated rectangle (no bullseye nearby) is not mistaken for a glyph", () => {
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    cv.rectangle(sheet, new cv.Point(200, 200), new cv.Point(600, 500), new cv.Scalar(0), 5);
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 0);
  });

  console.log(`\n${pass} passed`);
};
