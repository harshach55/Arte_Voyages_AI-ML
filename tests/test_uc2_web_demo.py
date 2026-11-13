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


# --- rules + trained tagger -----------------------------------------------------------------------


class FakeTagger:
    """Marks the word Zanzibar as a destination and "just me" as the party."""

    def labels_for(self, language):
        return {"DESTINATION", "PARTY"}

    def predict(self, words, language):
        tags = ["O"] * len(words)
        for i, word in enumerate(words):
            if word == "Zanzibar":
                tags[i] = "B-DESTINATION"
            elif word == "just" and words[i + 1:i + 2] == ["me"]:
                tags[i:i + 2] = ["B-PARTY", "I-PARTY"]
        return tags, [0.82] * len(words)


TAGGED = "We would like to see Zanzibar, it is just me from the office."


@pytest.fixture
def tagger_on(monkeypatch):
    from src.uc2_extractor import tagger

    monkeypatch.setattr(tagger, "default_tagger", lambda: FakeTagger())


def test_page_offers_the_third_mode_and_can_show_the_crf_tag():
    page = demo.render_page(demo.SAMPLES, False, "x", True, "Trained tagger available.").decode("utf-8")
    assert 'value="tagger"' in page and "rules + trained tagger" in page
    assert '"tag crf"' in page and "Trained tagger available." in page


def test_page_shows_every_pipeline_stage_in_order():
    page = demo.render_page(demo.SAMPLES, False, "x", True, "y").decode("utf-8")
    assert ">Pipeline<" in page
    stages = ["Language detection", "Trained tagger", "Rules and normalisers", "Optional embedding model",
              "Schema output", "Review screen"]
    assert [s["name"] for s in demo.PIPELINE] == stages
    assert all(f'"{s}"' in page for s in stages)


def test_components_say_what_is_trained_rules_and_pending():
    kinds = {c["kind"] for c in demo.components(True)}
    assert {"trained", "rules", "pending"} <= kinds
    assert demo.components(True)[0]["name"] == "Slot tagger (CRF)"
    assert all(c["kind"] in demo.KIND_NAMES for c in demo.PIPELINE + demo.components(True))


def test_a_missing_tagger_shows_as_pending_and_its_mode_is_off():
    assert demo.components(False)[0]["kind"] == "pending"
    page = demo.render_page(demo.SAMPLES, False, "x").decode("utf-8")
    assert '"available": false' in page


def test_tagger_values_show_the_crf_tag(tagger_on):
    data = demo.build_response(TAGGED, False, True)
    row = next(r for r in data["rows"] if r["label"] == "Destination")
    assert row["items"] == [{"text": "Zanzibar", "source": "crf", "confidence": 0.82}]
    assert row["evidence"] == [{"text": "Zanzibar", "source": "crf"}]
    travellers = next(r for r in data["rows"] if r["label"] == "Travellers")
    assert travellers["items"][0]["source"] == "crf" and travellers["items"][0]["confidence"] == 0.82


def test_a_section_with_rule_and_tagger_values_shows_both(tagger_on):
    data = demo.build_response("It is just me and 1 child this time", False, True)
    travellers = next(r for r in data["rows"] if r["label"] == "Travellers")
    assert travellers["items"][0]["source"] == "rule+crf"


def test_rules_only_is_unchanged_when_the_tagger_is_off(tagger_on):
    assert demo.build_response(TAGGED, False)["values"]["destination"] == ""


def test_batch_marks_tagger_sections_in_the_text(tagger_on):
    row = demo.build_batch([TAGGED], False, True)[0]
    assert row["destination"][0]["source"] == "crf"
    assert row["travellers"].endswith("(crf)")


def test_saved_correction_recomputes_with_the_tagger(tagger_on, tmp_path):
    path = tmp_path / "corrections.jsonl"
    values = demo.build_response(TAGGED, False, True)["values"]
    status, _ = demo.handle_save({"text": TAGGED, "mode": "tagger", "corrected": values}, False, path, True)
    assert status == 200
    saved = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "Zanzibar" in json.dumps(saved)


# --- new tagger, model comparison and export --------------------------------------------------------


def test_page_offers_the_new_tagger_comparison_and_export():
    page = demo.render_page(demo.SAMPLES, False, "x", True, "y", True, "New tagger available.").decode("utf-8")
    assert 'value="tagger2"' in page and "rules + new tagger" in page
    assert ">Model comparison<" in page and 'id="comparison"' in page
    for button in ("single-json", "single-csv", "batch-json", "batch-csv"):
        assert f'id="{button}"' in page
    assert "createObjectURL" in page and "innerHTML" not in page


def test_components_list_both_taggers():
    names = [c["name"] for c in demo.components(True, True)]
    assert names[:2] == ["Slot tagger (CRF)", "Slot tagger v2 (CRF, public + synthetic)"]
    assert demo.components(True, False)[1]["kind"] == "pending"


def test_handler_only_uses_a_tagger_that_is_loaded():
    handler = object.__new__(demo.Handler)
    handler.tagger_available, handler.tagger2_available = True, False
    assert handler.tagger_choice("v1") is True and handler.tagger_choice("v2") is False
    assert handler.tagger_choice(None) is False


def test_batch_rows_carry_all_values_for_export():
    row = demo.build_batch(["Two weeks in Kerala in March"], False)[0]
    from src.uc2_extractor.metrics import ALL_FIELDS

    assert list(row["values"]) == list(ALL_FIELDS) and row["values"]["destination"] == "Kerala"


def test_comparison_is_read_from_the_file_not_typed_in(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "MODEL_DIR", tmp_path)
    assert demo.load_comparison() is None
    (tmp_path / "comparison_v2.json").write_text('{"x": 1}', encoding="utf-8")
    assert demo.load_comparison() == {"x": 1}


def test_page_script_has_no_raw_line_breaks_inside_strings():
    page = demo.render_page(demo.SAMPLES, False, "x", True, "y", True, "z").decode("utf-8")
    script = page.split("<script>", 1)[1]
    assert "\r" not in script
    assert 'join("\\r\\n")' in script  # the CSV line break stays an escape in the page


def test_page_offers_the_field_selective_mode():
    page = demo.render_page(demo.SAMPLES, False, "x", True, "y", True, "z", True).decode("utf-8")
    assert 'value="tagger2s"' in page and "rules + new tagger (field-selective)" in page


def test_handler_field_selective_choice_needs_its_list():
    handler = object.__new__(demo.Handler)
    handler.tagger_available, handler.tagger2_available, handler.tagger2s_available = True, True, False
    assert handler.tagger_choice("v2_selective") is False
    handler.tagger2s_available = True
    assert handler.tagger_choice("v2_selective") == "v2_selective"
