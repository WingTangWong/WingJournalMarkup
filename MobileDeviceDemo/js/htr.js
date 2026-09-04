/* Handwriting-to-text — a second, small OCR backend alongside Tesseract.js
 * (js/ocr.js). Tesseract is an LSTM trained mostly on printed fonts and is
 * weak on cursive (per its own module doc); this is a CRNN+CTC model trained
 * on real handwriting (IAM database), run through onnxruntime-web (WASM).
 *
 * Model provenance / license — please read before changing or redistributing:
 *   Harald Scheidl's browser HTR demo (https://githubharald.github.io/text_reader.html),
 *   itself "a scaled-down version of the CRNN model used in the HTRPipeline
 *   repository" (https://github.com/githubharald/HTRPipeline), trained on the
 *   IAM Handwriting Database (CC BY-NC-SA 4.0 — non-commercial). Vendored here
 *   with attribution; keep this **non-commercial** and keep the attribution if
 *   you fork this. No redistribution license is stated by the author beyond
 *   the public demo itself, so treat this as a research/hobby-project
 *   inclusion, not a cleared-for-commercial-use asset (spec: none — this is a
 *   demo-only concern, see MobileDeviceDemo/README.md).
 *
 * Constraints inherited from that model (real ones, not tuning knobs):
 *   - fixed 256x48 input, image is *stretched* to fit (no letterpad) — matches
 *     exactly how the int8 quantization was calibrated, so don't "improve" the
 *     preprocessing without recalibrating the model;
 *   - 42-symbol charset: digits, A-Z, ÄÖÜ, ' and - — UPPERCASE ONLY, no
 *     lowercase, no punctuation beyond apostrophe/hyphen;
 *   - one word or a short phrase per call, not a full sentence.
 * That makes it a good fit for this project's metadata fields (short,
 * conventionally-caps values like "#RESEARCH") and a poor fit for body
 * prose — see how js/app.js calls it (metadata cells only, alongside
 * Tesseract for comparison, not instead of it).
 *
 * recognize(canvas) -> { text, words:[{text,bbox,confidence}], backend, recognized }
 * — same shape as js/ocr.js's recognizers, so the two can be swapped/compared.
 */

const CHARS = [
  "'", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "Ä", "Ö", "Ü", "’",
];
const WIDTH = 256;
const HEIGHT = 48;
const SCALE_DOWN = 8;
const NUM_CHARS = CHARS.length + 1; // + CTC blank
const NUM_TIMESTEPS = WIDTH / SCALE_DOWN;

const UNRECOGNIZED = "�";
const unreadable = (backend) => ({ text: UNRECOGNIZED, words: [], backend, recognized: false });

function injectScript(url) {
  return new Promise((resolve, reject) => {
    if (window.ort) return resolve();
    const s = document.createElement("script");
    s.src = url;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("failed to load " + url));
    document.head.appendChild(s);
  });
}

// stretch (not letterboxed — matches how the int8 model was calibrated) the
// crop into WIDTHxHEIGHT grayscale, normalized like the reference JS impl
function tensorFromCanvas(canvas) {
  const work = document.createElement("canvas");
  work.width = WIDTH;
  work.height = HEIGHT;
  const ctx = work.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.drawImage(canvas, 0, 0, WIDTH, HEIGHT);
  const data = ctx.getImageData(0, 0, WIDTH, HEIGHT).data;

  const arr = new Float32Array(WIDTH * HEIGHT);
  for (let i = 0; i < arr.length; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    arr[i] = gray / 255 - 0.5;
  }
  return arr;
}

// CTC best-path (greedy) decode: argmax per timestep, collapse repeats, drop
// blanks (class 0). Mirrors htr_pipeline.reader.ctc.ctc_best_path.
function ctcBestPath(predictions) {
  let text = "";
  let prob = 1;
  let prev = -1;
  for (let t = 0; t < NUM_TIMESTEPS; t++) {
    let bestP = -Infinity;
    let bestC = 0;
    for (let c = 0; c < NUM_CHARS; c++) {
      const p = predictions[t * NUM_CHARS + c];
      if (p > bestP) { bestP = p; bestC = c; }
    }
    prob *= bestP;
    if (bestC !== 0 && bestC !== prev) text += CHARS[bestC - 1];
    prev = bestC;
  }
  // per-timestep geometric mean, so length doesn't dominate the score
  const confidence = NUM_TIMESTEPS ? Math.pow(Math.max(prob, 0), 1 / NUM_TIMESTEPS) : 0;
  return { text, confidence: Math.round(confidence * 1000) / 1000 };
}

export class HTRRecognizer {
  /** @param {{ortJs, wasm, model}} paths — vendor/htr/{ort.min.js,ort-wasm-simd.wasm,model_int8.onnx} */
  constructor(paths) {
    this.name = "htr";
    this.paths = paths;
    this.session = null;
  }

  async init() {
    await injectScript(this.paths.ortJs);
    const ort = window.ort;
    if (!ort) throw new Error("onnxruntime-web did not load");
    // pin to the one vendored (SIMD, single-threaded) wasm binary regardless
    // of which filename the runtime's auto-detection asks for
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      "ort-wasm.wasm": this.paths.wasm,
      "ort-wasm-simd.wasm": this.paths.wasm,
    };
    this.session = await ort.InferenceSession.create(this.paths.model, {
      executionProviders: ["wasm"],
    });
    return this;
  }

  /** @param {HTMLCanvasElement} canvas a single word / short-phrase crop */
  async recognize(canvas) {
    if (!this.session) return unreadable(this.name);
    let out;
    try {
      const input = tensorFromCanvas(canvas);
      const ort = window.ort;
      const feeds = { input: new ort.Tensor("float32", input, [1, 1, HEIGHT, WIDTH]) };
      const results = await this.session.run(feeds);
      out = results[this.session.outputNames[0]].data;
    } catch (e) {
      return unreadable(this.name);
    }
    const { text, confidence } = ctcBestPath(out);
    if (!text) return { text: "", words: [], backend: this.name, recognized: false };
    return {
      text,
      words: [{ text, bbox: [0, 0, canvas.width, canvas.height], confidence }],
      backend: this.name,
      recognized: true,
    };
  }

  async terminate() {
    if (this.session) { try { await this.session.release(); } catch (e) { /* */ } this.session = null; }
  }
}

/** Resolves to an HTRRecognizer, or null if the runtime/model failed to load. */
export async function getHTR(paths) {
  try {
    return await new HTRRecognizer(paths).init();
  } catch (e) {
    return null;
  }
}
