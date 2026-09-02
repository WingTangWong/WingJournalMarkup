"""``wingjournal`` command-line entry point.

Implemented now:
    ingest          run the ingestion pipeline on an image or directory
    make-sheet      write a printable blank writing sheet (PDF)
    make-legend     write a printable WJM markup legend (PDF)
    make-test-page  write a synthetic WJM page image (flat or perspective-warped)
    eval            score boundary + orientation detection on a synthetic corpus
    dictionaries    list available ArUco dictionaries
    version

Stubs (see docs/ROADMAP.md):
    show-page, history
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wingjournal import __version__

# Mirror of wingjournal.vision.aruco.DEFAULT_DICT, kept here so `--help` and
# `version` do not have to import OpenCV. test_cli guards the two against drift.
_DEFAULT_DICT = "DICT_4X4_50"


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"wingjournal {__version__}")
    return 0


def _cmd_dictionaries(_args: argparse.Namespace) -> int:
    from wingjournal.vision.aruco import available_dictionaries

    for name in available_dictionaries():
        print(name)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from wingjournal.pipeline import ingest_path
    from wingjournal.vision.hypothesis import ScoringWeights

    weights = ScoringWeights.load(args.weights) if args.weights else None
    results = ingest_path(
        args.path, args.out, dict_name=args.dict, recursive=args.recursive,
        weights=weights, debug=args.debug,
    )
    if not results:
        print("no images found", file=sys.stderr)
        return 1
    for r in results:
        cap = r.capture
        print(
            f"{r.name}: {len(cap.detected_fiducials)} marker(s) + "
            f"{len(r.square_candidates)} square candidate(s), "
            f"boundary={cap.page_boundary_method} (score {cap.page_boundary_confidence:.2f}), "
            f"orientation={cap.orientation_degrees}deg/{cap.orientation_method} "
            f"-> {cap.normalized_image_path}"
        )
        for note in cap.notes:
            print(f"    - {note}")
    return 0


def _cmd_make_sheet(args: argparse.Namespace) -> int:
    from wingjournal.templates import build_writing_sheet

    out = build_writing_sheet(
        args.out, paper=args.paper, pages=args.pages, dpi=args.dpi,
        dict_name=args.dict, marker_mm=args.marker_mm, ruled=args.ruled,
    )
    print(f"wrote {out} ({args.pages} page(s), {args.paper}, {args.dpi} dpi)")
    return 0


def _cmd_make_legend(args: argparse.Namespace) -> int:
    from wingjournal.templates import build_legend_pdf

    out = build_legend_pdf(args.out, paper=args.paper, dpi=args.dpi)
    print(f"wrote {out}")
    return 0


def _cmd_make_test_page(args: argparse.Namespace) -> int:
    import cv2

    from wingjournal.vision.synthetic import make_page, warp_page

    page = make_page(dict_name=args.dict)
    out = Path(args.out)
    if args.warp:
        scene, _quad = warp_page(page, seed=args.seed)
        cv2.imwrite(str(out), scene)
    else:
        cv2.imwrite(str(out), page)
    print(f"wrote {out}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from wingjournal.eval import format_report, run_eval
    from wingjournal.vision.hypothesis import ScoringWeights

    weights = ScoringWeights.load(args.weights) if args.weights else None
    report = run_eval(n=args.cases, seed=args.seed, weights=weights)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, verbose=args.verbose))
    return 0


def _cmd_stub(name: str):
    def run(_args: argparse.Namespace) -> int:
        print(
            f"'{name}' is not implemented yet - it needs the page/document graph "
            f"(see docs/ROADMAP.md).",
            file=sys.stderr,
        )
        return 2

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wingjournal", description="Wing Journal Markup")
    parser.add_argument("--version", action="version", version=f"wingjournal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="run the ingestion pipeline on an image/directory")
    p_ingest.add_argument("path", help="image file or directory of images")
    p_ingest.add_argument("--out", default="out", help="output directory (default: ./out)")
    p_ingest.add_argument("--dict", default=_DEFAULT_DICT, help="ArUco dictionary name")
    p_ingest.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    p_ingest.add_argument("--weights", help="JSON file of hypothesis scoring weights")
    p_ingest.add_argument(
        "--debug", action="store_true", help="write vision overlays to <out>/debug"
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_sheet = sub.add_parser("make-sheet", help="printable blank WJM writing sheet (PDF or image)")
    p_sheet.add_argument(
        "--out", default="wjm-writing-sheet.pdf",
        help="output path; .pdf for a (multi-page) PDF, .png/.jpg for a raster",
    )
    p_sheet.add_argument("--paper", default="letter", choices=("letter", "a4", "legal"))
    p_sheet.add_argument("--pages", type=int, default=1, help="number of sheets")
    p_sheet.add_argument("--dpi", type=int, default=300)
    p_sheet.add_argument("--dict", default=_DEFAULT_DICT, help="ArUco dictionary name")
    p_sheet.add_argument("--marker-mm", type=float, default=18.0, help="marker size in mm")
    p_sheet.add_argument("--ruled", action="store_true", help="faint ruled lines in the body")
    p_sheet.set_defaults(func=_cmd_make_sheet)

    p_legend = sub.add_parser("make-legend", help="printable WJM markup legend (PDF)")
    p_legend.add_argument("--out", default="wjm-legend.pdf", help="output PDF path")
    p_legend.add_argument("--paper", default="letter", choices=("letter", "a4", "legal"))
    p_legend.add_argument("--dpi", type=int, default=200)
    p_legend.set_defaults(func=_cmd_make_legend)

    p_make = sub.add_parser("make-test-page", help="synthetic WJM page image for testing")
    p_make.add_argument("--out", default="test-page.png", help="output image path")
    p_make.add_argument("--dict", default=_DEFAULT_DICT, help="ArUco dictionary name")
    p_make.add_argument("--warp", action="store_true", help="apply a random perspective")
    p_make.add_argument("--seed", type=int, default=None, help="RNG seed for --warp")
    p_make.set_defaults(func=_cmd_make_test_page)

    p_eval = sub.add_parser("eval", help="score boundary + orientation on a synthetic corpus")
    p_eval.add_argument("--cases", type=int, default=24, help="number of corpus cases")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--weights", help="JSON file of hypothesis scoring weights")
    p_eval.add_argument("--json", action="store_true", help="emit the full report as JSON")
    p_eval.add_argument("--verbose", action="store_true", help="per-case rows")
    p_eval.set_defaults(func=_cmd_eval)

    p_dict = sub.add_parser("dictionaries", help="list available ArUco dictionaries")
    p_dict.set_defaults(func=_cmd_dictionaries)

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=_cmd_version)

    for name in ("show-page", "history"):
        p = sub.add_parser(name, help=f"[stub] {name}")
        p.add_argument("ref", nargs="?", help="page reference, e.g. Research:P017")
        p.set_defaults(func=_cmd_stub(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
