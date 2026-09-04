#!/usr/bin/env bash
#
# Vendor the handwriting-recognition assets: onnxruntime-web (WASM runtime)
# and Harald Scheidl's int8-quantized CRNN+CTC model. See js/htr.js's header
# for what this model is, its constraints, and its (non-commercial-leaning)
# provenance before you use this outside a research/hobby context.
#
#   ./fetch-htr.sh              # pinned onnxruntime-web version below
#
# onnxruntime-web >= 1.19 only ships large "threaded" WASM builds (13-28 MB);
# 1.18.0 is the newest release that still has a small single-threaded SIMD
# build (~10 MB), which is what a plain GitHub Pages deploy (no COOP/COEP
# headers, so no SharedArrayBuffer/threads anyway) actually needs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ORT_VERSION="1.18.0"
OUT="vendor/htr"
mkdir -p "$OUT"

fetch() {
    local url="$1" out="$2" min_bytes="$3"
    echo "downloading ${url}"
    curl -fL --progress-bar "$url" -o "$out"
    local bytes
    bytes=$(wc -c < "$out")
    if [ "$bytes" -lt "$min_bytes" ]; then
        echo "!!! ${out} is only ${bytes} bytes — download likely failed" >&2
        exit 1
    fi
    echo "wrote ${out} (${bytes} bytes)"
}

fetch "https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js" \
    "$OUT/ort.min.js" 100000
fetch "https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort-wasm-simd.wasm" \
    "$OUT/ort-wasm-simd.wasm" 5000000
fetch "https://raw.githubusercontent.com/githubharald/githubharald.github.io/master/text_reader_ort/model_int8.onnx" \
    "$OUT/model_int8.onnx" 500000

echo "ok — reload the page; js/htr.js will pick these up from vendor/htr/."
