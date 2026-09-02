# Normalized page coordinates

Everything downstream of perspective normalization works in **normalized page
coordinates**. This document pins them so repeated captures of one physical page
are directly comparable (spec §35, §37; resolves ASSESSMENT §3.10).

## The frame

The page frame is the **marker constellation** — the outer corner of each of the
four corner fiducials — ordered `TL → TR → BR → BL` (spec §7). When markers are
absent the best boundary hypothesis stands in for it. The frame is *not* the
physical paper edge; the paper extends slightly beyond it.

## Axes and origin

- Origin `(0, 0)` at the top-left corner of the frame.
- `+x` points right, `+y` points **down** (image convention).
- The page is rectified **upright**: orientation resolution runs first and the
  rotation is folded into the stored homography, so `(0, 0)` is the top-left of
  the page as a reader holds it.

## Units and size

Normalized coordinates are **pixels at a fixed target size**, not the observed
capture resolution. `rectify()` scales the frame so its **longer side is
`target_long_px` pixels** (default **1600**, ≈145 DPI for a Letter page),
preserving the observed aspect ratio (clamped to `[1.15, 1.6]` — the plausible
range for Letter 1.29 and A4 1.41). A capture taken close-up and one taken far
away therefore produce normalized images of the same size, so pixel coordinates,
bounding boxes, and image diffs line up across captures.

Higher-resolution work (OCR in M4) can re-rectify from the stored raw image at a
larger `target_long_px` without changing the coordinate *convention*.

## The stored transform

`Capture.homography` is the 3×3 matrix mapping **raw-image** coordinates to the
**saved normalized image** (upright rotation included). To place a raw-image
point in normalized space, apply the homography with a perspective divide. Its
inverse maps normalized coordinates back onto the original capture.

## Provenance

`Capture.page_boundary_method` / `…_confidence` and `Capture.orientation_*`
record how the frame and the upright rotation were derived, and
`Capture.page_hypotheses` keeps the alternatives. Normalized coordinates are only
as trustworthy as the frame they came from — always carry the confidence.
