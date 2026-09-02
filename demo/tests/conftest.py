import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WJM_DEMO_DATA", str(tmp_path / "data"))
    # app reads the env at import; import fresh per test
    for mod in [m for m in list(sys.modules) if m.startswith("wjm_demo")]:
        del sys.modules[mod]
    from wjm_demo.app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


@pytest.fixture
def page_png():
    import cv2

    from wingjournal.vision.synthetic import make_page, warp_page

    scene, _ = warp_page(make_page(literal_box=True), seed=3)
    return cv2.imencode(".png", scene)[1].tobytes()
