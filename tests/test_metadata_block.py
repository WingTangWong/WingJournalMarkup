import cv2

from wingjournal.recognition.metadata_block import detect_metadata_block
from wingjournal.templates.geometry import compute_layout
from wingjournal.templates.writing_sheet import render_writing_sheet
from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.orientation import resolve_orientation
from wingjournal.vision.preprocess import preprocess
from wingjournal.vision.synthetic import warp_page


def test_detect_block_on_writing_sheet():
    sheet = render_writing_sheet(compute_layout(dpi=200))
    mb = detect_metadata_block(sheet, markers=detect_markers(sheet))
    assert mb is not None
    assert len(mb.row1_cells) == 3
    assert len(mb.row2_cells) == 4
    assert mb.confidence >= 0.9
    # located from the per-field ArUco anchors, not the thin rules
    assert mb.detection == "field_anchors"
    assert set(mb.field_cells) == {
        "document_id", "page_id", "topic_tags", "left", "above", "below", "right",
    }
    # block sits near the top of the page
    assert mb.bbox[1] < 0.35 * compute_layout(dpi=200).height_px


def test_falls_back_to_ruled_lines_without_marks():
    import numpy as np

    # a plain ruled 2-row grid, no registration marks
    img = np.full((900, 1400), 255, np.uint8)
    x0, y0, x1, y1 = 120, 60, 1280, 150
    my = (y0 + y1) // 2
    span = x1 - x0
    cv2.rectangle(img, (x0, y0), (x1, y1), 0, 3)
    cv2.line(img, (x0, my), (x1, my), 0, 3)
    for f in (1 / 3, 2 / 3):
        cx = int(x0 + span * f)
        cv2.line(img, (cx, y0), (cx, my), 0, 3)
    for f in (1 / 4, 2 / 4, 3 / 4):
        cx = int(x0 + span * f)
        cv2.line(img, (cx, my), (cx, y1), 0, 3)
    mb = detect_metadata_block(img)
    assert mb is not None
    assert mb.detection == "ruled_lines"
    assert (len(mb.row1_cells), len(mb.row2_cells)) == (3, 4)


def test_detect_block_survives_the_pipeline():
    from wingjournal.pipeline import ingest_image

    scene, _ = warp_page(render_writing_sheet(compute_layout(dpi=200)), seed=3)
    result = ingest_image("s", scene)
    mb = result.capture.metadata_block
    assert mb is not None
    assert mb["detection"] == "field_anchors"
    assert len(mb["row1_cells"]) == 3
    # sharpness assessed and recorded (spec §9.x)
    assert result.capture.sharpness is not None
    assert result.capture.sharpness["blurry"] is False


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
