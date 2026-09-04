"""Smoke test against REALWORLD-SAMPLES/ — real phone photos (real
handwriting, lighting, rotation, wrinkling), at three resolution tiers. The
folder is git-ignored and personal, so this skips cleanly everywhere it
isn't present (fresh clones, CI). See REALWORLD-SAMPLES/README.md.

This is deliberately not a strict accuracy test: there's no hand-labeled
ground truth for these yet, and the point right now is visibility into how
the pipeline (and, eventually, the demo's OCR/HTR backends) behaves on real
input, not a pass/fail bar. Run with -v -s to see the per-sample findings:

    .venv/bin/pytest tests/test_realworld_samples.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent / "REALWORLD-SAMPLES"
TIERS = ["orig-hires", "scaled-1080p", "scaled-720p"]

pytestmark = pytest.mark.skipif(
    not ROOT.is_dir(), reason="REALWORLD-SAMPLES/ not present (personal, git-ignored)"
)


def _samples() -> list[str]:
    hi = ROOT / "orig-hires"
    if not hi.is_dir():
        return []
    return sorted(p.stem for p in hi.glob("misc-*.jpg"))


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


SAMPLES = _samples()
CASES = [(name, tier) for name in SAMPLES for tier in TIERS]


@pytest.mark.skipif(not CASES, reason="no misc-*.jpg under REALWORLD-SAMPLES/orig-hires/")
@pytest.mark.parametrize("name,tier", CASES, ids=[f"{n}/{t}" for n, t in CASES])
def test_realworld_sample(name: str, tier: str) -> None:
    from wingjournal.pipeline import ingest_image

    path = ROOT / tier / f"{name}.jpg"
    if not path.is_file():
        pytest.skip(f"{path} missing (add it alongside the other tiers)")

    image = cv2.imread(str(path))
    assert image is not None, f"cv2 could not read {path}"

    recognizer = "auto" if _tesseract_available() else "none"
    result = ingest_image(f"{name}/{tier}", image, recognizer=recognizer, parse_body=True)
    cap = result.capture

    print(f"\n--- {name} [{tier}] {image.shape[1]}x{image.shape[0]} ---")
    print(f"  boundary: {cap.page_boundary_method} (score {cap.page_boundary_confidence:.2f})")
    print(f"  orientation: {cap.orientation_degrees} deg via {cap.orientation_method}")
    print(f"  sharpness: {cap.sharpness}")
    if cap.metadata_block:
        mb = cap.metadata_block
        print(f"  metadata block via {mb['detection']} (conf {mb['confidence']})")
    md = cap.page_metadata or {}
    print(f"  document_id={md.get('document_id')!r} page_id={md.get('page_id')!r} "
          f"topic_tags={md.get('topic_tags')!r}")
    print(f"  left={md.get('left')!r} above={md.get('above')!r} "
          f"below={md.get('below')!r} right={md.get('right')!r}")
    print(f"  {len(cap.detected_elements)} body element(s), "
          f"{len(cap.literal_assets)} literal region(s)")
    for note in cap.notes:
        print(f"  - {note}")

    # smoke-level only: the pipeline must not crash and must produce a capture
    # with *some* boundary, however low-confidence, for every tier
    assert cap.page_boundary_method is not None
