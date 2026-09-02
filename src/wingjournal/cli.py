"""``wingjournal`` command-line entry point.

    ingest          run the ingestion pipeline on an image or directory
                    (--store persists captures to a WJM store)
    show-page       show a stored page's metadata, relationships, conflicts
    history         show a stored page's capture timeline
    make-sheet      write a printable blank writing sheet (PDF or PNG)
    make-legend     write a printable WJM markup legend (PDF)
    make-test-page  write a synthetic WJM page image (flat or perspective-warped)
    eval            score boundary + orientation detection on a synthetic corpus
    dictionaries    list available ArUco dictionaries
    version
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
    store = None
    if args.store:
        from wingjournal.storage import Store

        store = Store(args.store)
    try:
        results = ingest_path(
            args.path, args.out, dict_name=args.dict, recursive=args.recursive,
            weights=weights, debug=args.debug, store=store, recognizer=args.recognizer,
        )
    finally:
        if store is not None:
            store.close()
    if not results:
        print("no images found", file=sys.stderr)
        return 1
    for r in results:
        cap = r.capture
        page = f" page={cap.page_uuid[:8]}" if cap.page_uuid else ""
        print(
            f"{r.name}: {len(cap.detected_fiducials)} marker(s) + "
            f"{len(r.square_candidates)} square candidate(s), "
            f"boundary={cap.page_boundary_method} (score {cap.page_boundary_confidence:.2f}), "
            f"orientation={cap.orientation_degrees}deg/{cap.orientation_method}{page} "
            f"-> {cap.normalized_image_path}"
        )
        for note in cap.notes:
            print(f"    - {note}")
        md = cap.page_metadata
        if md and any(md.get(k) for k in ("document_id", "page_id", "topic_tags")):
            print(f"    - metadata (via {cap.text_backend}): doc={md.get('document_id')} "
                  f"page={md.get('page_id')} topics={md.get('topic_tags')}")
    return 0


def _open_existing_store(path: str):
    """Open a store only if it already exists; returns None (and prints) otherwise."""

    from pathlib import Path

    if not (Path(path) / "wjm.sqlite").is_file():
        print(f"no WJM store at {path!r} (run `wingjournal ingest --store {path}` first)",
              file=sys.stderr)
        return None
    from wingjournal.storage import Store

    return Store(path)


def _resolve_page(store, ref: str):
    """Accept a page uuid (or 8-char prefix) or an explicit/machine page id."""

    page = store.get_page(ref)
    if page is None:
        page = store.find_page(page_id_explicit=ref, page_id_machine=ref)
    if page is None:
        import sqlite3

        try:
            row = store.db.execute(
                "SELECT uuid FROM pages WHERE uuid LIKE ? LIMIT 2", (ref + "%",)
            ).fetchall()
            if len(row) == 1:
                page = store.get_page(row[0]["uuid"])
        except sqlite3.Error:
            pass
    return page


def _cmd_show_page(args: argparse.Namespace) -> int:
    if not args.ref:
        print("usage: wingjournal show-page <uuid|page-id> [--store DIR]", file=sys.stderr)
        return 2
    store = _open_existing_store(args.store)
    if store is None:
        return 1
    with store:
        page = _resolve_page(store, args.ref)
        if page is None:
            print(f"no page matching {args.ref!r} in {args.store}", file=sys.stderr)
            return 1
        print(f"page {page.uuid}")
        print(f"  document : explicit={page.document_id_explicit} "
              f"resolved={page.document_id_resolved} ({page.document_id_resolution_source})")
        print(f"  page id  : handwritten={page.page_id_explicit} machine={page.page_id_machine}")
        print(f"  topics   : {', '.join(page.topic_tags) or '-'}")
        print(f"  neighbours: L={page.left} A={page.above} B={page.below} R={page.right}")
        print(f"  captures : {len(page.capture_uuids)}")
        rels = store.relationships_for_page(page.uuid)
        for rel in rels:
            arrow = "->" if rel.source_page == page.uuid else "<-"
            other = rel.target_page if rel.source_page == page.uuid else rel.source_page
            kind = "explicit" if rel.explicitly_declared else "inferred"
            print(f"  rel      : {arrow} {rel.relation} {other[:8]} ({kind})")
        conflicts = store.conflicts(page.uuid)
        for c in conflicts:
            print(f"  CONFLICT : [{c.kind}] {c.detail}")
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    if not args.ref:
        print("usage: wingjournal history <uuid|page-id> [--store DIR]", file=sys.stderr)
        return 2
    store = _open_existing_store(args.store)
    if store is None:
        return 1
    with store:
        page = _resolve_page(store, args.ref)
        if page is None:
            print(f"no page matching {args.ref!r} in {args.store}", file=sys.stderr)
            return 1
        caps = store.captures_for_page(page.uuid)
        print(f"page {page.uuid}: {len(caps)} capture(s)")
        for i, c in enumerate(caps, 1):
            print(
                f"  {i:>3}. {c['timestamp']}  {c.get('source_type', '?'):9} "
                f"boundary={c.get('page_boundary_method')} "
                f"({c.get('page_boundary_confidence', 0):.2f})  "
                f"orient={c.get('orientation_degrees')}deg  "
                f"norm_blob={(c.get('normalized_blob') or '')[:12]}"
            )
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
    p_ingest.add_argument("--store", help="persist captures into this WJM store directory")
    p_ingest.add_argument(
        "--recognizer", default="auto", choices=("auto", "tesseract", "none"),
        help="text recognizer for metadata cells (auto: use tesseract if installed)",
    )
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

    p_show = sub.add_parser("show-page", help="show a stored page's metadata + captures")
    p_show.add_argument("ref", nargs="?", help="page uuid (or prefix) or page id")
    p_show.add_argument("--store", default="wjm-store", help="WJM store directory")
    p_show.set_defaults(func=_cmd_show_page)

    p_hist = sub.add_parser("history", help="show a stored page's capture timeline")
    p_hist.add_argument("ref", nargs="?", help="page uuid (or prefix) or page id")
    p_hist.add_argument("--store", default="wjm-store", help="WJM store directory")
    p_hist.set_defaults(func=_cmd_history)

    p_dict = sub.add_parser("dictionaries", help="list available ArUco dictionaries")
    p_dict.set_defaults(func=_cmd_dictionaries)

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
