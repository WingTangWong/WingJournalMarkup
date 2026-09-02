import json

import cv2
import numpy as np

from wingjournal.pipeline import ingest_image, ingest_path
from wingjournal.vision.synthetic import make_page, warp_page


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def test_ingest_image_rectifies_warped_page(warped_scene):
    scene, _quad = warped_scene
    result = ingest_image("scene", scene)

    cap = result.capture
    assert len(cap.detected_fiducials) == 4
    assert cap.page_boundary_method == "aruco_constellation"
    assert cap.page_boundary_confidence >= 0.9

    homography = np.array(cap.homography)
    assert homography.shape == (3, 3)

    # The detected boundary must map onto the corners of the normalized image.
    norm = result.normalized_image
    h, w = norm.shape[:2]
    quad = np.array(cap.page_boundary_polygon, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(quad, homography).reshape(-1, 2)
    expected = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    np.testing.assert_allclose(mapped, expected, atol=1.0)

    # Normalized page keeps the source orientation (portrait).
    assert h > w


def test_ingest_path_writes_sidecar(tmp_path, warped_scene):
    scene, _quad = warped_scene
    img_path = tmp_path / "page01.png"
    cv2.imwrite(str(img_path), scene)

    out_dir = tmp_path / "out"
    results = ingest_path(img_path, out_dir)
    assert len(results) == 1

    norm = out_dir / "normalized" / "page01.png"
    sidecar = out_dir / "captures" / "page01.json"
    assert norm.is_file() and sidecar.is_file()

    data = json.loads(sidecar.read_text())
    assert data["page_boundary_method"] == "aruco_constellation"
    assert len(data["detected_fiducials"]) == 4
    assert data["normalized_image_path"].endswith("normalized/page01.png")


def test_ingest_directory(tmp_path, flat_page):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"p{i}.png"), flat_page)
    results = ingest_path(tmp_path, tmp_path / "out")
    assert len(results) == 3


def test_homography_includes_upright_rotation():
    # a 90-degrees-CW page: the stored homography must fold in the upright
    # rotation, so it maps the raw boundary onto the *saved* (rotated) image.
    page = cv2.rotate(make_page(), cv2.ROTATE_90_CLOCKWISE)
    scene, _ = warp_page(page, seed=9)
    result = ingest_image("rot", scene)
    cap = result.capture

    assert cap.orientation_degrees == 270  # undo 90 CW

    norm = result.normalized_image
    h, w = norm.shape[:2]
    quad = np.array(cap.page_boundary_polygon, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(quad, np.array(cap.homography)).reshape(-1, 2)
    # rotation permutes which corner is where, so compare as an unordered set
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    np.testing.assert_allclose(_sorted_rows(mapped), _sorted_rows(corners), atol=1.5)


def test_recursive_ingest_disambiguates_same_stem(tmp_path, flat_page):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        cv2.imwrite(str(tmp_path / sub / "page.png"), flat_page)

    results = ingest_path(tmp_path, tmp_path / "out", recursive=True)
    names = sorted(r.name for r in results)
    assert names == ["a__page", "b__page"]
    for n in names:
        assert (tmp_path / "out" / "normalized" / f"{n}.png").is_file()


def test_no_fiducials_falls_back(tmp_path):
    blank = np.full((800, 600, 3), 255, dtype=np.uint8)
    result = ingest_image("blank", blank)
    assert result.capture.detected_fiducials == []
    assert result.capture.page_boundary_method in {"largest_quad", "full_frame"}
    assert any("no fiducial evidence" in n for n in result.capture.notes)
