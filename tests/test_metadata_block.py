import cv2

from wingjournal.recognition.metadata_block import detect_metadata_block
from wingjournal.templates.geometry import compute_layout
from wingjournal.templates.writing_sheet import render_writing_sheet
from wingjournal.vision.orientation import resolve_orientation
from wingjournal.vision.preprocess import preprocess
from wingjournal.vision.synthetic import warp_page


def test_detect_block_on_writing_sheet():
    mb = detect_metadata_block(render_writing_sheet(compute_layout(dpi=200)))
    assert mb is not None
    assert len(mb.row1_cells) == 3
    assert len(mb.row2_cells) == 4
    assert mb.confidence >= 0.9
    # block sits near the top of the page
    assert mb.bbox[1] < 0.35 * compute_layout(dpi=200).height_px


def test_detect_block_survives_the_pipeline():
    from wingjournal.pipeline import ingest_image

    scene, _ = warp_page(render_writing_sheet(compute_layout(dpi=200)), seed=3)
    result = ingest_image("s", scene)
    assert result.capture.metadata_block is not None
    assert len(result.capture.metadata_block["row1_cells"]) == 3


def test_no_block_on_blank_image():
    import numpy as np

    assert detect_metadata_block(np.full((800, 600, 3), 255, np.uint8)) is None


def test_metadata_block_resolves_orientation_without_markers():
    # an upside-down markerless page: the block fixes the 180 flip that the
    # text-baseline tier can't. Feed a clean rectified page (no perspective).
    rectified = cv2.rotate(
        render_writing_sheet(compute_layout(dpi=150)), cv2.ROTATE_180
    )
    o = resolve_orientation(preprocess(rectified), [], rectified=rectified)
    assert o.method == "metadata_block"
    assert o.degrees == 180
    assert o.flip_ambiguous is False
