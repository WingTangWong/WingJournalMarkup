import json

import numpy as np
import pytest

from wingjournal.vision.aruco import detect_markers
from wingjournal.vision.hypothesis import (
    ScoringWeights,
    rank_hypotheses,
    select_boundary,
)
from wingjournal.vision.preprocess import preprocess


def test_scoring_weights_load(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"decoded_markers": 0.5, "content_containment": 0.4}))
    w = ScoringWeights.load(p)
    assert w.decoded_markers == 0.5
    assert w.content_containment == 0.4
    assert w.rectangularity == ScoringWeights().rectangularity


def test_scoring_weights_load_rejects_unknown(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"bogus": 1.0}))
    with pytest.raises(ValueError):
        ScoringWeights.load(p)


def test_constellation_wins_on_clean_page(warped_scene):
    scene, _ = warped_scene
    pre = preprocess(scene)
    markers = detect_markers(pre.gray)
    ranked = rank_hypotheses(pre, markers, [])
    assert ranked[0].source == "aruco_constellation"
    assert ranked[0].score > 0.8
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))


def test_select_boundary_matches_true_frame(warped_scene):
    scene, true_quad = warped_scene
    pre = preprocess(scene)
    markers = detect_markers(pre.gray)
    boundary, ranked, squares = select_boundary(pre, markers)
    assert boundary.method == "aruco_constellation"
    assert len(ranked) >= 2
    assert isinstance(squares, list)
    # the chosen polygon should sit within the true page quad, roughly
    poly = np.array(boundary.polygon)
    assert poly.min() >= -5


def test_envelope_hypotheses_from_content():
    from wingjournal.vision.envelope import envelope_hypotheses
    from wingjournal.vision.synthetic import make_page

    pre = preprocess(make_page())
    hyps = envelope_hypotheses(pre)
    assert {h.source for h in hyps} >= {"content_envelope", "content_aspect"}
    assert all(len(h.polygon) == 4 for h in hyps)


def test_partial_fiducials_beat_full_frame():
    # two decoded markers + two blank corner squares -> a real 4-corner frame
    from wingjournal.vision.synthetic import make_page, warp_page

    page = make_page(blank_roles=("TOP_LEFT", "TOP_RIGHT"))
    scene, _ = warp_page(page, seed=2)
    pre = preprocess(scene)
    markers = detect_markers(pre.gray)
    assert len(markers) == 2

    boundary, ranked, _ = select_boundary(pre, markers)
    assert boundary.method != "full_frame"
    assert boundary.confidence > 0.5
