"""Run the ingestion front-end over the corpus and score it."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from wingjournal.eval.corpus import EvalCase, generate_corpus
from wingjournal.eval.metrics import polygon_iou
from wingjournal.pipeline import ingest_image
from wingjournal.vision.hypothesis import ScoringWeights


@dataclass
class CaseResult:
    name: str
    markers_detected: int
    dropped_markers: int
    perspective: str
    boundary_iou: float
    boundary_method: str
    orientation_ok: bool
    true_orientation: int
    predicted_orientation: int


@dataclass
class Bucket:
    label: str
    n: int
    mean_iou: float
    orientation_accuracy: float


@dataclass
class EvalReport:
    n_cases: int
    mean_iou: float
    orientation_accuracy: float
    buckets: list[Bucket] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "mean_iou": self.mean_iou,
            "orientation_accuracy": self.orientation_accuracy,
            "buckets": [asdict(b) for b in self.buckets],
            "cases": [asdict(c) for c in self.cases],
        }


def _evaluate_case(case: EvalCase, weights: ScoringWeights | None) -> CaseResult:
    result = ingest_image(case.name, case.image, weights=weights)
    cap = result.capture
    iou = polygon_iou(cap.page_boundary_polygon, case.true_polygon, case.image.shape[:2])
    pred = cap.orientation_degrees or 0
    return CaseResult(
        name=case.name,
        markers_detected=len(cap.detected_fiducials),
        dropped_markers=case.dropped_markers,
        perspective=case.perspective,
        boundary_iou=round(iou, 4),
        boundary_method=cap.page_boundary_method or "?",
        orientation_ok=(pred == case.true_orientation),
        true_orientation=case.true_orientation,
        predicted_orientation=pred,
    )


def run_eval(
    cases: list[EvalCase] | None = None,
    n: int = 24,
    seed: int = 0,
    weights: ScoringWeights | None = None,
) -> EvalReport:
    cases = cases or generate_corpus(n=n, seed=seed)
    rows = [_evaluate_case(c, weights) for c in cases]

    by_markers: dict[int, list[CaseResult]] = {}
    for r in rows:
        by_markers.setdefault(r.markers_detected, []).append(r)

    buckets = [
        Bucket(
            label=f"{k} markers",
            n=len(v),
            mean_iou=round(statistics.fmean(x.boundary_iou for x in v), 4),
            orientation_accuracy=round(
                statistics.fmean(1.0 if x.orientation_ok else 0.0 for x in v), 4
            ),
        )
        for k, v in sorted(by_markers.items(), reverse=True)
    ]

    return EvalReport(
        n_cases=len(rows),
        mean_iou=round(statistics.fmean(r.boundary_iou for r in rows), 4),
        orientation_accuracy=round(
            statistics.fmean(1.0 if r.orientation_ok else 0.0 for r in rows), 4
        ),
        buckets=buckets,
        cases=rows,
    )


def format_report(report: EvalReport, verbose: bool = False) -> str:
    lines = [
        f"cases: {report.n_cases}",
        f"boundary IoU (mean): {report.mean_iou:.3f}",
        f"orientation accuracy: {report.orientation_accuracy:.3f}",
        "",
        f"{'bucket':<12} {'n':>3} {'mean IoU':>10} {'orient acc':>11}",
        "-" * 40,
    ]
    for b in report.buckets:
        lines.append(f"{b.label:<12} {b.n:>3} {b.mean_iou:>10.3f} {b.orientation_accuracy:>11.3f}")
    if verbose:
        lines += ["", f"{'case':<10} {'mk':>3} {'IoU':>7} {'method':<20} {'orient':>14}"]
        lines.append("-" * 60)
        for c in report.cases:
            orient = f"{c.predicted_orientation}/{c.true_orientation}"
            flag = "" if c.orientation_ok else " x"
            lines.append(
                f"{c.name:<10} {c.markers_detected:>3} {c.boundary_iou:>7.3f} "
                f"{c.boundary_method:<20} {orient:>12}{flag}"
            )
    return "\n".join(lines)
