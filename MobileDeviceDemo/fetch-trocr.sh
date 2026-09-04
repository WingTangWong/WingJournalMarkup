#!/usr/bin/env bash
#
# Vendor the heavier handwriting-recognition assets: Transformers.js (bundles
# a matching onnxruntime-web) and Xenova's int8-quantized ONNX conversion of
# TrOCR-small-handwritten. See js/trocr.js's header for what this model is,
# its (~76 MB, ~5-10s/field) cost, and its non-commercial-leaning provenance
# before using this outside a research/hobby context.
#
#   ./fetch-trocr.sh
#
# Pinned to @xenova/transformers@2.17.2, whose dist/ bundles onnxruntime-web
# 1.14.0 directly (the last release with a small, non-threaded WASM build —
# see fetch-htr.sh) — so the JS and the one WASM file it needs come from the
# exact same place, no separate version-matching to get right.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
TRANSFORMERS_VERSION="2.17.2"
HF_MODEL="Xenova/trocr-small-handwritten"
OUT="vendor/trocr"
MODEL_DIR="$OUT/models/$HF_MODEL"
mkdir -p "$OUT" "$MODEL_DIR/onnx"

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

fetch "https://cdn.jsdelivr.net/npm/@xenova/transformers@${TRANSFORMERS_VERSION}/dist/transformers.min.js" \
    "$OUT/transformers.min.js" 500000
fetch "https://cdn.jsdelivr.net/npm/@xenova/transformers@${TRANSFORMERS_VERSION}/dist/ort-wasm-simd.wasm" \
    "$OUT/ort-wasm-simd.wasm" 5000000

HF="https://huggingface.co/${HF_MODEL}/resolve/main"
for f in config.json generation_config.json preprocessor_config.json \
         tokenizer_config.json special_tokens_map.json tokenizer.json \
         sentencepiece.bpe.model; do
    fetch "$HF/$f" "$MODEL_DIR/$f" 100
done
fetch "$HF/onnx/encoder_model_quantized.onnx" "$MODEL_DIR/onnx/encoder_model_quantized.onnx" 10000000
fetch "$HF/onnx/decoder_model_merged_quantized.onnx" "$MODEL_DIR/onnx/decoder_model_merged_quantized.onnx" 20000000

echo "ok — reload the page; js/trocr.js will pick these up from vendor/trocr/."
