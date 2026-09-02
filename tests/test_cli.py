
from wingjournal.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "wingjournal" in capsys.readouterr().out


def test_dictionaries(capsys):
    assert main(["dictionaries"]) == 0
    assert "DICT_4X4_50" in capsys.readouterr().out


def test_make_test_page_and_ingest(tmp_path, capsys):
    page = tmp_path / "tp.png"
    assert main(["make-test-page", "--out", str(page), "--warp", "--seed", "7"]) == 0
    assert page.is_file()

    out = tmp_path / "out"
    assert main(["ingest", str(page), "--out", str(out)]) == 0
    captured = capsys.readouterr().out
    assert "aruco_constellation" in captured
    assert (out / "normalized" / "tp.png").is_file()


def test_ingest_debug_writes_overlays(tmp_path):
    page = tmp_path / "tp.png"
    main(["make-test-page", "--out", str(page), "--warp", "--seed", "3"])
    out = tmp_path / "out"
    assert main(["ingest", str(page), "--out", str(out), "--debug"]) == 0
    overlays = sorted(p.name for p in (out / "debug").glob("*.png"))
    assert overlays == [
        "tp.01_markers.png", "tp.02_candidates.png",
        "tp.03_hypotheses.png", "tp.04_chosen.png",
    ]


def test_make_sheet_command(tmp_path):
    out = tmp_path / "sheet.pdf"
    assert main(["make-sheet", "--out", str(out), "--pages", "2", "--dpi", "120"]) == 0
    assert out.read_bytes()[:5] == b"%PDF-"


def test_make_legend_command(tmp_path):
    out = tmp_path / "legend.pdf"
    assert main(["make-legend", "--out", str(out), "--dpi", "120"]) == 0
    assert out.is_file()


def test_eval_command_json(capsys):
    assert main(["eval", "--cases", "6", "--seed", "0", "--json"]) == 0
    import json

    report = json.loads(capsys.readouterr().out)
    assert report["n_cases"] == 6
    assert "buckets" in report


def test_cli_default_dict_matches_aruco():
    from wingjournal.cli import _DEFAULT_DICT
    from wingjournal.vision.aruco import DEFAULT_DICT

    assert _DEFAULT_DICT == DEFAULT_DICT


def test_show_page_without_store_errors(tmp_path):
    assert main(["show-page", "Research:P017", "--store", str(tmp_path / "nope")]) == 1


def test_ingest_store_then_show_and_history(tmp_path, capsys):
    page = tmp_path / "p.png"
    main(["make-test-page", "--out", str(page), "--warp", "--seed", "5"])
    store = tmp_path / "store"

    assert main(["ingest", str(page), "--out", str(tmp_path / "out"),
                 "--store", str(store)]) == 0
    assert (store / "wjm.sqlite").is_file()
    out = capsys.readouterr().out
    assert "page=" in out

    import sqlite3

    db = sqlite3.connect(store / "wjm.sqlite")
    page_uuid = db.execute("SELECT uuid FROM pages").fetchone()[0]
    db.close()

    assert main(["show-page", page_uuid[:8], "--store", str(store)]) == 0
    assert "captures : 1" in capsys.readouterr().out

    assert main(["history", page_uuid, "--store", str(store)]) == 0
    assert "1 capture(s)" in capsys.readouterr().out
