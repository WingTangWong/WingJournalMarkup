import numpy as np
import pytest

from wingjournal.vision.synthetic import make_page, warp_page


@pytest.fixture
def flat_page() -> np.ndarray:
    return make_page()


@pytest.fixture
def warped_scene():
    page = make_page()
    scene, quad = warp_page(page, seed=1234)
    return scene, quad
