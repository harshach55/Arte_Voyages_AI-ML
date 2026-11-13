"""Test for the home page: numbers come from the results files, and a missing file says so."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("home", ROOT / "app" / "home.py")
home = importlib.util.module_from_spec(spec)
spec.loader.exec_module(home)
NO_API = "http://127.0.0.1:9/openapi.json"  # nothing listens here


def test_home_numbers_come_from_the_results_files_and_missing_files_say_so(tmp_path):
    missing = home.headline_numbers(tmp_path, NO_API)
    assert set(missing) == {"extractor", "highlighter", "rates", "api"}
    assert {n["value"] for card in missing.values() for n in card} == {"not available"}

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "uc2_results.md").write_text(
        "- **The only fair number is the first held-out run** (0.11 / 0.22 / 0.33). Later numbers are optimistic.",
        encoding="utf-8")
    crf = tmp_path / "models" / "uc2_crf"
    crf.mkdir(parents=True)
    (crf / "comparison_v2.json").write_text(json.dumps({"heldout": {"errors": {"rules only": 7, "rules + v2": 9}}}),
                                            encoding="utf-8")
    (crf / "crf_tagger.json").write_text(json.dumps({"dev_report": {
        lang: {"ALL": {"f1": f1}} for lang, f1 in (("en", 0.5), ("fr", 0.25), ("de", 0.125))}}), encoding="utf-8")
    rates = tmp_path / "models" / "uc21_rates"
    rates.mkdir(parents=True)
    (rates / "metrics.json").write_text(json.dumps({"overall": {"gradient_boosting": {"mae": 12.345},
                                                                "baseline": {"mae": 20.0}}}), encoding="utf-8")
    numbers = home.headline_numbers(tmp_path, NO_API)
    values = {card: [n["value"] for n in rows] for card, rows in numbers.items()}
    assert values["extractor"] == ["en 0.11 / fr 0.22 / de 0.33", "7 / 9"]
    assert values["highlighter"] == ["en 0.50 / fr 0.25 / de 0.12"]
    assert values["rates"] == ["12.35 / 20.00 EUR", "not available"]  # calibration file still missing
    assert values["api"] == ["not available"]  # no API running at that address

    page = home.render_page(numbers).decode("utf-8")
    assert "innerHTML" not in page and "document.write" not in page
    assert all(f'"port": {c["port"]}' in page for c in home.CARDS)
    assert '"C2"' in page and "planned" in page and 'mode: "no-cors"' in page
