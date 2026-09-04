#!/usr/bin/env bash
#
# Vendor a local copy of OpenCV.js so the demo loads fast and works offline.
# The app falls back to the docs.opencv.org CDN when vendor/opencv.js is absent.
#
#   ./fetch-opencv.sh              # latest 4.x build
#   ./fetch-opencv.sh 4.10.0       # a specific version
#
# The docs.opencv.org build embeds the .wasm as a data: URI, so this one file is
# the whole engine. This demo only needs core imgproc (threshold, findContours,
# approxPolyDP) — no ArUco/objdetect bindings required, unlike MobileDeviceDemo.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
VERSION="${1:-4.x}"
URL="https://docs.opencv.org/${VERSION}/opencv.js"
OUT="vendor/opencv.js"

mkdir -p vendor
echo "downloading ${URL}"
curl -fL --progress-bar "$URL" -o "$OUT"

bytes=$(wc -c < "$OUT")
echo "wrote ${OUT} (${bytes} bytes)"
if [ "$bytes" -lt 1000000 ]; then
    echo "!!! that looks too small — the URL may be wrong for version '${VERSION}'" >&2
    exit 1
fi
if ! grep -q "wasmBinaryFile" "$OUT"; then
    echo "!!! no 'wasmBinaryFile' marker — not an OpenCV.js WASM build" >&2
    exit 1
fi
echo "ok — reload the page; it will use the vendored copy."
