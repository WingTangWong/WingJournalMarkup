/* Smoke test for js/vision-core.js against the vendored OpenCV.js (WASM).
 * Generates a synthetic sheet (4 DICT_4X4_50 markers + a ruled 2-row box +
 * text lines) and checks the geometric detectors.
 *
 *   node tests/vision.test.mjs        (needs vendor/opencv.js)
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
  require(path.join(root, "js/vision-core.js"));
  const V = globalThis.WJMVision;

  // --- synthetic sheet: 900x1160 gray, markers 96px at 60px inset -------
  const W = 900, H = 1160, SZ = 96, M = 60;
  const sheet = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(255));
  const dict = cv.getPredefinedDictionary(cv.DICT_4X4_50);
  const place = { 0: [M, M], 1: [W - M - SZ, M], 2: [W - M - SZ, H - M - SZ], 3: [M, H - M - SZ] };
  for (const [id, [x, y]] of Object.entries(place)) {
    const mk = new cv.Mat();
    cv.aruco_generateImageMarker
      ? cv.aruco_generateImageMarker(dict, +id, SZ, mk, 1)
      : cv.generateImageMarker(dict, +id, SZ, mk, 1);
    mk.copyTo(sheet.roi(new cv.Rect(x, y, SZ, SZ)));
    mk.delete();
  }
  // a ruled 2-row metadata box between the top markers
  const bx = M + SZ + 16, bw = W - 2 * bx, by = M + 6, bh = 84;
  cv.rectangle(sheet, new cv.Point(bx, by), new cv.Point(bx + bw, by + bh), new cv.Scalar(0), 2);
  cv.line(sheet, new cv.Point(bx, by + bh / 2), new cv.Point(bx + bw, by + bh / 2), new cv.Scalar(0), 2);
  for (const f of [1 / 3, 2 / 3]) cv.line(sheet, new cv.Point(bx + bw * f, by), new cv.Point(bx + bw * f, by + bh / 2), new cv.Scalar(0), 2);
  for (const f of [1 / 4, 2 / 4, 3 / 4]) cv.line(sheet, new cv.Point(bx + bw * f, by + bh / 2), new cv.Point(bx + bw * f, by + bh), new cv.Scalar(0), 2);
  // concentric-square registration marks at the four block corners (on a moat)
  const REG = 26;
  for (const [cx, cy] of [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]) {
    const sq = (frac, val) => {
      const r = Math.round(REG * frac / 2);
      cv.rectangle(sheet, new cv.Point(cx - r, cy - r), new cv.Point(cx + r, cy + r), new cv.Scalar(val), -1);
    };
    sq(1.45, 255); sq(1.0, 0); sq(0.5, 255); sq(0.22, 0);
  }
  // a few body "text" lines
  for (let i = 0; i < 3; i++) cv.putText(sheet, "the quick brown fox", new cv.Point(120, 360 + i * 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0), 2);

  t("MarkerDetector finds all four ids with roles", () => {
    const det = new V.MarkerDetector(cv);
    const markers = det.detect(sheet);
    assert.deepEqual(markers.map((m) => m.id), [0, 1, 2, 3]);
    assert.deepEqual(markers.map((m) => m.role),
      ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"]);
  });

  t("pageQuadFromMarkers -> ordered quad near the marker outer corners", () => {
    const det = new V.MarkerDetector(cv);
    const quad = V.pageQuadFromMarkers(det.detect(sheet));
    assert.equal(quad.length, 4);
    assert.ok(Math.abs(quad[0][0] - M) < 4 && Math.abs(quad[0][1] - M) < 4, "TL ~ (60,60)");
    assert.ok(Math.abs(quad[2][0] - (W - M)) < 4, "BR x ~ W-60");
  });

  t("rectify -> 1600 long side default, plausible aspect", () => {
    const det = new V.MarkerDetector(cv);
    const quad = V.pageQuadFromMarkers(det.detect(sheet));
    const { mat, width, height } = V.rectify(cv, sheet, quad);
    assert.equal(Math.max(width, height), 1600);
    const ar = Math.max(width, height) / Math.min(width, height);
    assert.ok(ar >= 1.15 && ar <= 1.6, "aspect " + ar);
    mat.delete();
  });

  t("rectify -> honours an explicit targetLong", () => {
    const det = new V.MarkerDetector(cv);
    const quad = V.pageQuadFromMarkers(det.detect(sheet));
    const { mat, width, height } = V.rectify(cv, sheet, quad, 2800);
    assert.equal(Math.max(width, height), 2800);
    mat.delete();
  });

  t("rectify -> fixedSize gives an exact canvas (8.5x11 @ 300)", () => {
    const det = new V.MarkerDetector(cv);
    const quad = V.pageQuadFromMarkers(det.detect(sheet));
    const { mat, width, height } = V.rectify(cv, sheet, quad, null, [2550, 3300]);
    assert.equal(width, 2550);
    assert.equal(height, 3300);
    mat.delete();
  });

  t("tenengrad -> sharp scene scores higher than a blurred one", () => {
    const blur = new cv.Mat();
    cv.GaussianBlur(sheet, blur, new cv.Size(0, 0), 3);
    const sharpScore = V.tenengrad(cv, sheet);
    const blurScore = V.tenengrad(cv, blur);
    assert.ok(sharpScore > blurScore * 1.5, `${sharpScore} vs ${blurScore}`);
    blur.delete();
  });

  t("detectRegistrationMarks -> the four corner marks", () => {
    const marks = V.detectRegistrationMarks(cv, sheet, [0, 0, W, 260], null);
    assert.equal(marks.length, 4, "got " + marks.length);
    assert.ok(marks.every((m) => m.rings >= 2 && m.acutance > 0.6));
    assert.ok(V.marksToQuad(marks) !== null);
  });

  t("detectMetadataBlock -> registration_marks path, 3 + 4 cells", () => {
    const block = V.detectMetadataBlock(cv, sheet);
    assert.ok(block, "block found");
    assert.equal(block.detection, "registration_marks");
    assert.equal(block.registration_marks.length, 4);
    assert.equal(block.row1_cells.length, 3);
    assert.equal(block.row2_cells.length, 4);
    assert.ok(block.confidence >= 0.8);
  });

  t("assessSharpness -> crisp render is not blurry; blurred one is", () => {
    const det = new V.MarkerDetector(cv);
    const marks = V.detectRegistrationMarks(cv, sheet, [0, 0, W, 260], null);
    const crisp = V.assessSharpness(cv, sheet, det.detect(sheet), marks);
    assert.ok(crisp.score > 0.6 && !crisp.blurry, JSON.stringify(crisp));
    const soft = new cv.Mat();
    cv.GaussianBlur(sheet, soft, new cv.Size(0, 0), 4);
    const rep = V.assessSharpness(cv, soft, det.detect(soft), []);
    assert.ok(rep.blurry, JSON.stringify(rep));
    soft.delete();
  });

  t("segmentLines -> finds the body lines", () => {
    const lines = V.segmentLines(cv, sheet);
    assert.ok(lines.length >= 3, "got " + lines.length + " lines");
    for (const l of lines) assert.equal(l.length, 4);
  });

  t("detectLiteralAssets -> none on a sheet with no corner-fill boxes", () => {
    assert.deepEqual(V.detectLiteralAssets(cv, sheet), []);
  });

  t("corner stickers -> roles, quad, page-size estimate", () => {
    // a blank page with a synthetic id-10 sticker (wedge + bracket + aruco) at
    // each corner, rotated for its corner
    const PW = 700, PH = 900;
    const page = new cv.Mat(PH, PW, cv.CV_8UC1, new cv.Scalar(255));
    const SS = 96;
    const one = new cv.Mat(SS, SS, cv.CV_8UC1, new cv.Scalar(255));
    // wedge (solid triangle) at TL
    const tri = cv.matFromArray(3, 1, cv.CV_32SC2, [0, 0, 30, 0, 0, 30]);
    const tv = new cv.MatVector(); tv.push_back(tri);
    cv.fillPoly(one, tv, new cv.Scalar(0));
    cv.rectangle(one, new cv.Point(0, 0), new cv.Point(46, 3), new cv.Scalar(0), -1);
    cv.rectangle(one, new cv.Point(0, 0), new cv.Point(3, 46), new cv.Scalar(0), -1);
    const mk = new cv.Mat();
    (cv.aruco_generateImageMarker || cv.generateImageMarker)(dict, 10, 44, mk, 1);
    mk.copyTo(one.roi(new cv.Rect(26, 26, 44, 44)));

    const place = (rot, x, y) => {
      const r = new cv.Mat();
      if (rot) cv.rotate(one, r, rot); else one.copyTo(r);
      r.copyTo(page.roi(new cv.Rect(x, y, SS, SS)));
      r.delete();
    };
    place(null, 3, 3);
    place(cv.ROTATE_90_CLOCKWISE, PW - SS - 3, 3);
    place(cv.ROTATE_180, PW - SS - 3, PH - SS - 3);
    place(cv.ROTATE_90_COUNTERCLOCKWISE, 3, PH - SS - 3);

    const st = new V.MarkerDetector(cv).detect(page).filter((m) => m.id === V.CORNER_STICKER_ID);
    assert.equal(st.length, 4, "sticker markers " + st.length);
    const stickers = V.detectCornerStickers(cv, page, st);
    assert.deepEqual(
      new Set(stickers.map((s) => s.role)),
      new Set(["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"]),
    );
    assert.ok(V.stickerQuad(stickers) !== null);
    const est = V.estimatePageSize(stickers);
    assert.ok(est && est.width_mm > 0 && est.height_mm > est.width_mm);

    page.delete(); one.delete(); mk.delete(); tri.delete(); tv.delete();
  });

  sheet.delete();
  console.log(`\n${pass} passed`);
  process.exit(process.exitCode || 0);
};
