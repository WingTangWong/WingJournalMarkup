/* Text recognition — the main-thread mirror of
 *   src/wingjournal/recognition/text/__init__.py  (TextRecognizer / NullRecognizer)
 *   src/wingjournal/recognition/text/tesseract.py (TesseractRecognizer)
 *
 * Tesseract.js manages its own Web Worker, so calling it from the main thread
 * still keeps OCR off the UI thread. It needs a canvas it cannot get inside our
 * vision worker, which is why recognition lives here.
 *
 *   recognize(canvas) -> {
 *     text, words:[{text,bbox:[x,y,w,h],confidence}], backend, recognized
 *   }
 *
 * Recognition is always optional: with no engine every region comes back
 * unrecognized and the pipeline still completes (spec §48).
 */

const UNRECOGNIZED = "�";
const unreadable = (backend) => ({ text: UNRECOGNIZED, words: [], backend, recognized: false });

function injectScript(url) {
  return new Promise((resolve, reject) => {
    if (window.Tesseract) return resolve();
    const s = document.createElement("script");
    s.src = url;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("failed to load " + url));
    document.head.appendChild(s);
  });
}

export class NullRecognizer {
  constructor() { this.name = "none"; }
  async recognize() { return unreadable("none"); }
  async terminate() {}
}

export class TesseractRecognizer {
  /** @param {{tesseractJs,workerPath,corePath,langPath}} paths */
  constructor(paths) {
    this.name = "tesseract";
    this.paths = paths;
    this.worker = null;
  }

  async init() {
    await injectScript(this.paths.tesseractJs);
    const w = await window.Tesseract.createWorker("eng", 1 /* OEM.LSTM_ONLY */, {
      workerPath: this.paths.workerPath,
      corePath: this.paths.corePath,
      langPath: this.paths.langPath,
      gzip: true,
    });
    await w.setParameters({ tessedit_pageseg_mode: "6" }); // uniform block of text
    this.worker = w;
    return this;
  }

  /** @param {HTMLCanvasElement} canvas */
  async recognize(canvas) {
    if (!this.worker) return unreadable(this.name);
    let res;
    try { res = await this.worker.recognize(canvas); }
    catch (e) { return unreadable(this.name); }
    const d = res.data || {};
    const words = (d.words || [])
      .filter((w) => w && w.text && w.text.trim() && w.confidence >= 0)
      .map((w) => ({
        text: w.text.trim(),
        bbox: [w.bbox.x0, w.bbox.y0, w.bbox.x1 - w.bbox.x0, w.bbox.y1 - w.bbox.y0],
        confidence: Math.round(w.confidence) / 100,
      }));
    return {
      text: words.map((w) => w.text).join(" "),
      words,
      backend: this.name,
      recognized: words.length > 0,
    };
  }

  async terminate() {
    if (this.worker) { try { await this.worker.terminate(); } catch (e) { /* */ } this.worker = null; }
  }
}

/** `auto`/`tesseract` -> Tesseract when its files load, else null. `none` -> null. */
export async function getRecognizer(prefer, paths) {
  if (prefer === "auto" || prefer === "tesseract") {
    try {
      return await new TesseractRecognizer(paths).init();
    } catch (e) {
      if (prefer === "tesseract") throw e;
    }
  }
  return new NullRecognizer();
}
