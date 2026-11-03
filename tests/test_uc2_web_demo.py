import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uc2_web_demo", ROOT / "app" / "uc2_web_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_model_missing_falls_back_to_rules_only(monkeypatch):
    from src.uc2_extractor import semantic

    def missing():
        raise FileNotFoundError("model files not found")

    monkeypatch.setattr(semantic, "default_embedder", missing)
    ok, message = demo.check_model()
    assert ok is False
    assert "rules only" in message


def test_page_never_inserts_html():
    page = demo.render_page(demo.SAMPLES, False, "rules only").decode("utf-8")
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page


def test_page_data_cannot_close_the_script_block():
    sample = [{"label": "x", "text": "</script><script>alert(1)</script>"}]
    page = demo.render_page(sample, False, "</script>").decode("utf-8")
    script = page.split("<script>", 1)[1]
    assert "</script><script>alert(1)" not in script


def test_scores_keep_real_and_optimistic_apart():
    assert demo.SCORES["real"]["title"].startswith("Real number")
    assert "optimistic" in demo.SCORES["optimistic"]["title"]
    assert demo.SCORES["real"]["rows"] == [["Rules only", "0.81", "0.88", "0.86"]]


def test_rows_tag_every_value_as_rule_in_rules_only_mode():
    data = demo.build_response("My wife and I want 10 nights in Kerala, vegetarian, budget USD 9,000.", False)
    labels = [r["label"] for r in data["rows"]]
    assert labels[0] == "Language" and "Special requests" in labels
    items = [it for r in data["rows"] for it in r["items"]]
    assert items and all(it["source"] == "rule" and it["confidence"] is None for it in items)
    json.dumps(data)  # the response must be valid JSON


def test_rows_show_model_values_with_confidence(monkeypatch):
    from src.uc2_extractor.schema import Provenance

    profile = demo.extract_profile("We love birds")
    profile.interests.append("wildlife")
    profile.provenance.append(Provenance(field="interests", value="wildlife", source="model", confidence=0.91))
    row = next(r for r in demo.build_rows(profile) if r["label"] == "Interests")
    assert row["items"] == [{"text": "wildlife", "source": "model", "confidence": 0.91}]


def test_batch_gives_one_summary_row_per_enquiry():
    rows = demo.build_batch(["Two weeks in Kerala in March", "Malediven, 10 Nächte, 2 Pers."], False)
    assert [r["n"] for r in rows] == [1, 2]
    assert rows[0]["destination"][0]["text"] == "Kerala"
    assert rows[1]["language"] == "de"
