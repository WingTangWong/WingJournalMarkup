"""Evaluation harness for boundary + orientation detection (M2 issue #13).

Generates a labelled synthetic corpus, runs the ingestion front-end over it, and
reports page-boundary IoU and orientation accuracy, bucketed by how many ArUco
markers were detected (mirrors the spec's confidence tiers, section 9).
"""

from wingjournal.eval.corpus import EvalCase, generate_corpus
from wingjournal.eval.harness import EvalReport, format_report, run_eval
from wingjournal.eval.metrics import polygon_iou

__all__ = [
    "EvalCase",
    "generate_corpus",
    "EvalReport",
    "format_report",
    "run_eval",
    "polygon_iou",
]
