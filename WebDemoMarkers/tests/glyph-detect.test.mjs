/* Smoke test for js/glyph-detect.js against the vendored OpenCV.js (WASM).
 * Draws synthetic bullseye+box+3-bit-code glyphs (mirroring the "Proposed
 * Handwritten Syntax Lexicon" sketch this detector is built from) and checks
 * that every bit pattern decodes correctly, including under rotation.
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

  // Draws one glyph: a box outline with a bullseye (ring+dot) just outside
  // its top-left corner, and 0/1 fills at the other three corners (bits in
  // {upperRight, lowerRight, lowerLeft} order — matches the UR/LR/LL walk).
  function drawGlyph(mat, bx, by, bw, bh, bits) {
    const black = new cv.Scalar(0);
    const white = new cv.Scalar(255);
    cv.rectangle(mat, new cv.Point(bx, by), new cv.Point(bx + bw, by + bh), black, 4);

    // bullseye: filled ring (outer disk minus inner disk) + center dot
    const cx = bx - 32, cy = by - 32;
    cv.circle(mat, new cv.Point(cx, cy), 18, black, -1);
    cv.circle(mat, new cv.Point(cx, cy), 11, white, -1);
    cv.circle(mat, new cv.Point(cx, cy), 5, black, -1);

    const S = 50; // corner fill square size
    if (bits.upperRight) cv.rectangle(mat, new cv.Point(bx + bw - S, by), new cv.Point(bx + bw, by + S), black, -1);
    if (bits.lowerRight) cv.rectangle(mat, new cv.Point(bx + bw - S, by + bh - S), new cv.Point(bx + bw, by + bh), black, -1);
    if (bits.lowerLeft) cv.rectangle(mat, new cv.Point(bx, by + bh - S), new cv.Point(bx + S, by + bh), black, -1);
  }

  const W = 900, H = 1160, BW = 160, BH = 200;
  const cases = [
    { value: 0, bits: { upperRight: 0, lowerRight: 0, lowerLeft: 0 } },
    { value: 5, bits: { upperRight: 1, lowerRight: 0, lowerLeft: 1 } },
    { value: 7, bits: { upperRight: 1, lowerRight: 1, lowerLeft: 1 } },
    { value: 2, bits: { upperRight: 0, lowerRight: 1, lowerLeft: 0 } },
  ];

  for (const { value, bits } of cases) {
    t(`decodes value ${value} (0b${value.toString(2).padStart(3, "0")})`, () => {
      const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
      drawGlyph(sheet, 300, 350, BW, BH, bits);
      const glyphs = G.detectGlyphs(cv, sheet);
      sheet.delete();
      assert.equal(glyphs.length, 1, `expected exactly one glyph, got ${glyphs.length}`);
      assert.equal(glyphs[0].value, value);
      assert.deepEqual(glyphs[0].bits, bits);
    });
  }

  t("survives a 25deg rotation (clockwise UR/LR/LL walk stays correct)", () => {
    const bits = { upperRight: 1, lowerRight: 0, lowerLeft: 1 }; // value 5
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    drawGlyph(sheet, 300, 350, BW, BH, bits);

    const center = new cv.Point(300 + BW / 2, 350 + BH / 2);
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
    drawGlyph(sheet, 300, 200, BW, BH, { upperRight: 1, lowerRight: 1, lowerLeft: 0 }); // 6
    drawGlyph(sheet, 300, 700, BW, BH, { upperRight: 0, lowerRight: 0, lowerLeft: 1 }); // 1
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 2, `expected two glyphs, got ${glyphs.length}`);
    const values = glyphs.map((g) => g.value).sort();
    assert.deepEqual(values, [1, 6]);
  });

  t("a blank page produces no glyphs", () => {
    const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
    const glyphs = G.detectGlyphs(cv, sheet);
    sheet.delete();
    assert.equal(glyphs.length, 0);
  });

  console.log(`\n${pass} passed`);
};
