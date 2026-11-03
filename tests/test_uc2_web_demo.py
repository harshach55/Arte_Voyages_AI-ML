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


# --- review and correct ---------------------------------------------------------------------


def test_page_shows_the_save_button_and_the_counter():
    page = demo.render_page(demo.SAMPLES, False, "rules only").decode("utf-8")
    assert "Save corrections" in page and 'id="correction-count"' in page


def test_correction_fields_match_the_gold_format():
    from src.uc2_extractor.metrics import ALL_FIELDS

    assert [f["name"] for f in demo.CORRECTION_FIELDS] == list(ALL_FIELDS)


def test_extract_response_includes_editable_values():
    data = demo.build_response("Two weeks in Kerala in March", False)
    assert data["values"]["destination"] == "Kerala"
    assert data["values"]["duration"] == "2 weeks"


def test_save_correction_writes_a_line_and_returns_the_count(tmp_path):
    path = tmp_path / "corrections.jsonl"
    text = "Two weeks in Kerala in March"
    values = demo.build_response(text, False)["values"]
    values["interests"] = "beach"
    status, body = demo.handle_save({"text": text, "mode": "rules", "corrected": values}, False, path)
    assert (status, body) == (200, {"count": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["corrected"]["interests"] == ["beach"]
    assert record["extracted"]["interests"] == []


def test_saved_extraction_is_recomputed_on_the_server(tmp_path):
    # whatever the browser sends, the 'extracted' part comes from the server
    path = tmp_path / "corrections.jsonl"
    text = "Two weeks in Kerala in March"
    values = demo.build_response(text, False)["values"]
    status, _ = demo.handle_save({"text": text, "mode": "rules", "corrected": values,
                                  "extracted": {"destination": ["Mars"]}}, False, path)
    assert status == 200
    assert json.loads(path.read_text(encoding="utf-8"))["extracted"]["destination"] == ["Kerala"]


@pytest.mark.parametrize(
    "data, message",
    [
        (None, "request must be JSON"),
        ({"text": "", "corrected": {}}, "empty or too long"),
        ({"text": "Kerala", "corrected": "x"}, "must map field names"),
        ({"text": "Kerala", "corrected": {"party_size": 3}}, "must map field names"),
        ({"text": "Kerala", "corrected": {"party_size": "three"}}, "Not saved"),
        ({"text": "Kerala", "corrected": {"hotel": "x"}}, "unknown fields"),
    ],
)
def test_bad_corrections_are_refused_and_not_saved(tmp_path, data, message):
    path = tmp_path / "corrections.jsonl"
    status, body = demo.handle_save(data, False, path)
    assert status == 400 and message in body["error"]
    assert not path.exists()
