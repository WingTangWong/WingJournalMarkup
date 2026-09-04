/* Handwriting-to-text, heavier model — TrOCR: a ViT image encoder + text
 * transformer decoder (Li et al., https://arxiv.org/abs/2109.10282),
 * fine-tuned on real handwriting. A proper attention-based OCR model, unlike
 * js/htr.js's small CRNN+CTC — tried after the CRNN's accuracy came back too
 * poor to be useful. Runs through Transformers.js (bundles onnxruntime-web).
 *
 * Model provenance / license — same situation as js/htr.js, please read:
 *   microsoft/trocr-small-handwritten (https://github.com/microsoft/unilm,
 *   code is MIT) is, like the CRNN, "fine-tuned on the IAM dataset"
 *   (CC BY-NC-SA 4.0 — non-commercial). No explicit redistribution license is
 *   stated for the checkpoint itself. The ONNX conversion vendored here is
 *   Xenova's (https://huggingface.co/Xenova/trocr-small-handwritten),
 *   int8-quantized. Vendored with attribution on the same non-commercial
 *   research/hobby understanding as the CRNN model — see MobileDeviceDemo/README.md.
 *
 * Much heavier than the CRNN: ~65 MB of ONNX weights + ~11 MB Transformers.js
 * runtime, and autoregressive decoding (slower than the CRNN's single forward
 * pass — expect roughly a second or more per field on a phone, not ~50ms).
 * Kept to short outputs (max_new_tokens) since these are metadata fields, not
 * paragraphs. No fixed input size or uppercase-only charset restriction like
 * the CRNN — this model reads mixed case and a full vocabulary.
 *
 * recognize(canvas) -> { text, words:[{text,bbox,confidence}], backend, recognized }
 * — same shape as js/ocr.js and js/htr.js, so all three can run side by side.
 */

const UNRECOGNIZED = "�";
const unreadable = (backend) => ({ text: UNRECOGNIZED, words: [], backend, recognized: false });

const MAX_NEW_TOKENS = 32; // metadata fields are short; bounds worst-case latency

// The DeiT image encoder resizes to a 384x384 SQUARE with no aspect
// preservation (preprocessor_config.json: do_resize + a plain size, no
// do_center_crop). A metadata-field crop is a wide, short strip (a page_id
// box can be 6-8:1 wide:tall) — handed to the model as-is, that resize
// vertically stretches the handwriting into scrambled noise, and the
// autoregressive decoder then free-associates fluent, entirely fabricated
// sentences from it (a language-model prior filling in for a garbage image
// signal) rather than failing loudly. Letterboxing onto a square canvas
// first — pad, don't stretch — keeps the glyphs' real proportions so the
// encoder sees actual handwriting shapes instead of taffy.
const SQUARE = 384;
function letterboxSquare(canvas) {
  const sq = document.createElement("canvas");
  sq.width = SQUARE;
  sq.height = SQUARE;
  const ctx = sq.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, SQUARE, SQUARE);
  const scale = Math.min(SQUARE / canvas.width, SQUARE / canvas.height);
  const w = Math.max(1, Math.round(canvas.width * scale));
  const h = Math.max(1, Math.round(canvas.height * scale));
  ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, (SQUARE - w) / 2, (SQUARE - h) / 2, w, h);
  return sq;
}

export class TrOCRRecognizer {
  /** @param {{transformersJs, modelsDir, wasmDir}} paths — see js/app.js's TROCR_PATHS */
  constructor(paths) {
    this.name = "trocr";
    this.paths = paths;
    this.pipe = null;
    this.RawImage = null;
  }

  async init() {
    const mod = await import(/* webpackIgnore: true */ this.paths.transformersJs);
    const { pipeline, env, RawImage } = mod;
    this.RawImage = RawImage;

    // serve everything from vendor/trocr/ — no calls out to huggingface.co
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = this.paths.modelsDir;
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.wasmPaths = this.paths.wasmDir;

    this.pipe = await pipeline("image-to-text", "Xenova/trocr-small-handwritten", {
      quantized: true,
    });
    return this;
  }

  /** @param {HTMLCanvasElement} canvas a single word / short-phrase crop */
  async recognize(canvas) {
    if (!this.pipe || !this.RawImage) return unreadable(this.name);
    let text = "";
    try {
      const squared = letterboxSquare(canvas);
      const ctx = squared.getContext("2d");
      const { data, width, height } = ctx.getImageData(0, 0, squared.width, squared.height);
      const img = new this.RawImage(data, width, height, 4);
      const out = await this.pipe(img, { max_new_tokens: MAX_NEW_TOKENS });
      text = ((out && out[0] && out[0].generated_text) || "").trim();
    } catch (e) {
      return unreadable(this.name);
    }
    if (!text) return { text: "", words: [], backend: this.name, recognized: false };
    return {
      text,
      // the pipeline gives one string for the whole crop, not per-word boxes/
      // confidence — report it as a single "word" spanning the crop, like
      // js/htr.js does, so it slots into the same debug-report code
      words: [{ text, bbox: [0, 0, canvas.width, canvas.height], confidence: null }],
      backend: this.name,
      recognized: true,
    };
  }

  async terminate() {
    this.pipe = null;
  }
}

/** Resolves to a TrOCRRecognizer, or null if the ~80 MB runtime/model failed
 * to load (never blocks — the caller just doesn't get a trocr reading). */
export async function getTrOCR(paths) {
  try {
    return await new TrOCRRecognizer(paths).init();
  } catch (e) {
    return null;
  }
}
