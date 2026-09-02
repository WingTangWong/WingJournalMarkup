import numpy as np

from wingjournal.eval import generate_corpus, polygon_iou, run_eval
from wingjournal.eval.harness import format_report


def test_polygon_iou_identical():
    quad = [[0, 0], [100, 0], [100, 80], [0, 80]]
    assert polygon_iou(quad, quad, (200, 200)) == 1.0


def test_polygon_iou_disjoint():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[50, 50], [60, 50], [60, 60], [50, 60]]
    assert polygon_iou(a, b, (100, 100)) == 0.0


def test_corpus_labels_are_sane():
    cases = generate_corpus(n=8, seed=1)
    assert len(cases) == 8
    for c in cases:
        assert np.array(c.true_polygon).shape == (4, 2)
        assert c.true_orientation in {0, 90, 180, 270}


def test_run_eval_buckets_meet_thresholds():
    report = run_eval(n=16, seed=0)
    assert report.n_cases == 16
    by_label = {b.label: b for b in report.buckets}

    four = by_label["4 markers"]
    assert four.mean_iou >= 0.95
    assert four.orientation_accuracy == 1.0

    # partial-fiducial cases must still land a usable frame, not fall to
    # full-frame (regression guard for the envelope / 3-corner work)
    for label in ("3 markers", "2 markers"):
        if label in by_label:
            assert by_label[label].mean_iou >= 0.65

    assert isinstance(format_report(report, verbose=True), str)
