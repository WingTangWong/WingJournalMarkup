
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


def test_stub_exits_2(capsys):
    assert main(["show-page", "Research:P017"]) == 2
