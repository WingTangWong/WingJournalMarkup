import numpy as np

from wingjournal.vision.literal_box import (
    crop_literal,
    detect_literal_assets,
    mask_literals,
)
from wingjournal.vision.synthetic import make_page, warp_page


def test_detects_the_four_corner_mount_box():
    page = make_page(literal_box=True)
    assets = detect_literal_assets(page)
    assert len(assets) == 1
    x, y, w, h = assets[0].bbox
    assert w > 0.5 * page.shape[1] * 0.5
    assert assets[0].confidence > 0.6


def test_no_false_positive_on_plain_page_or_blank():
    assert detect_literal_assets(make_page()) == []
    assert detect_literal_assets(np.full((900, 700, 3), 255, np.uint8)) == []


def test_survives_the_pipeline_and_is_masked():
    from wingjournal.pipeline import ingest_image

    scene, _ = warp_page(make_page(literal_box=True), seed=4)
    result = ingest_image("s", scene)
    assert len(result.capture.literal_assets) == 1
    assert any("literal image region" in n for n in result.capture.notes)


def test_mask_and_crop():
    page = make_page(literal_box=True)
    assets = detect_literal_assets(page)
    masked = mask_literals(page, assets)
    x, y, w, h = (int(v) for v in assets[0].bbox)
    centre = masked[y + h // 2, x + w // 2]
    assert (centre == 255).all()  # interior blanked

    crop = crop_literal(page, assets[0])
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] < h and crop.shape[1] < w


def test_persisted_literal_gets_a_blob(tmp_path):
    import cv2

    from wingjournal.pipeline import ingest_path
    from wingjournal.storage import Store

    img = tmp_path / "lit.png"
    cv2.imwrite(str(img), make_page(literal_box=True))
    with Store(tmp_path / "store") as store:
        results = ingest_path(img, tmp_path / "out", store=store)
    assets = results[0].capture.literal_assets
    assert len(assets) == 1
    assert assets[0]["asset_blob"] and len(assets[0]["asset_blob"]) == 64
