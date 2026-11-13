import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uc2_api", ROOT / "app" / "uc2_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

ENQUIRY = "My wife and I want 10 nights in Kerala in March 2027, vegetarian, budget USD 9,000."


@pytest.fixture
def rules_client():
    with TestClient(api.create_app(load_model=False)) as c:
        yield c


def fake_embed(texts):
    # birds -> dimension 0, everything else -> its own direction
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        if "bird" in t.lower() or "wild animals" in t.lower():
            out[i, 0] = 1.0
        else:
            out[i, 1 + (abs(hash(t)) % 63)] = 1.0
    return out


def test_health_without_model(rules_client):
    r = rules_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is False and body["threshold"] is None


def test_missing_model_files_fall_back_to_rules(monkeypatch):
    def missing():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(api.semantic, "default_embedder", missing)
    with TestClient(api.create_app(load_model=True)) as c:
        health = c.get("/health").json()
        assert health["model_loaded"] is False and "rules only" in health["model_message"]
        r = c.post("/extract", json={"text": ENQUIRY, "mode": "rules_model"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode_requested"] == "rules_model" and body["mode_used"] == "rules"
    assert "not available" in body["note"]


def test_extract_returns_a_validated_profile_with_evidence_flags_and_tags(rules_client):
    r = rules_client.post("/extract", json={"text": ENQUIRY})
    assert r.status_code == 200
    body = r.json()
    assert body["mode_used"] == "rules" and body["note"] is None
    profile = api.ClientProfile.model_validate(body["profile"])  # it must validate against the schema
    assert profile.destination == ["Kerala"]
    assert profile.dietary == ["vegetarian"]
    assert any(e.text == "Kerala" for e in profile.evidence)
    assert all(p.source == "rule" for p in profile.provenance)
    assert isinstance(body["profile"]["flags"], list)


def test_extract_with_model_marks_model_values():
    with TestClient(api.create_app(embedder=fake_embed)) as c:
        assert c.get("/health").json()["model_loaded"] is True
        r = c.post("/extract", json={"text": "He is mad about birds", "mode": "rules_model"})
    body = r.json()
    assert body["mode_used"] == "rules_model"
    tags = [p for p in body["profile"]["provenance"] if p["field"] == "interests"]
    assert tags and tags[0]["source"] == "model" and 0 < tags[0]["confidence"] <= 1


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"text": "   "}, "text text is empty"),
        ({"text": "a" * 20001}, "text must be at most 20000 characters"),
        ({"text": "Kerala", "mode": "fast"}, "mode must be 'rules', 'rules_model', 'rules_tagger' or 'rules_tagger_v2'"),
        ({"mode": "rules"}, "text is required"),
        ({"text": "Kerala", "colour": "red"}, "colour is not an accepted field"),
    ],
)
def test_bad_input_gives_short_clear_errors(rules_client, payload, expected):
    r = rules_client.post("/extract", json=payload)
    assert r.status_code == 422
    assert expected in r.json()["errors"]


def test_errors_never_repeat_the_enquiry_text(rules_client):
    secret = "Mr Example from 12 Sample Street"
    for payload in ({"text": secret, "mode": "fast"}, {"text": secret + " " * 2 + "a" * 20000}):
        r = rules_client.post("/extract", json=payload)
        assert r.status_code == 422
        assert secret not in r.text and "Sample Street" not in r.text


def test_body_that_is_not_json(rules_client):
    r = rules_client.post("/extract", content=b"hello", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "JSON object" in r.json()["errors"][0]


def test_very_large_body_is_refused(rules_client):
    r = rules_client.post("/extract", content=b"x" * 300_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_docs_page_and_schema_are_available(rules_client):
    assert rules_client.get("/docs").status_code == 200
    schema = rules_client.get("/openapi.json").json()
    assert "/extract" in schema["paths"] and "/health" in schema["paths"]


# --- rules_tagger -------------------------------------------------------------------------------


class FakeTagger:
    """Marks the word Zanzibar as a destination."""

    def labels_for(self, language):
        return {"DESTINATION"}

    def predict(self, words, language):
        tags = ["B-DESTINATION" if w == "Zanzibar" else "O" for w in words]
        return tags, [0.87] * len(words)


def test_rules_tagger_mode_marks_crf_values():
    with TestClient(api.create_app(load_model=False, crf_tagger=FakeTagger())) as c:
        assert c.get("/health").json()["tagger_loaded"] is True
        r = c.post("/extract", json={"text": "We would like to see Zanzibar with our kids", "mode": "rules_tagger"})
    body = r.json()
    assert r.status_code == 200 and body["mode_used"] == "rules_tagger" and body["note"] is None
    profile = api.ClientProfile.model_validate(body["profile"])
    assert profile.destination == ["Zanzibar"]
    tags = [p for p in profile.provenance if p.field == "destination"]
    assert tags[0].source == "crf" and tags[0].confidence == 0.87
    assert [e.source for e in profile.evidence if e.field == "destination"] == ["crf"]


def test_rules_tagger_mode_falls_back_to_rules_without_the_tagger():
    with TestClient(api.create_app(load_model=False, load_tagger=False)) as c:
        health = c.get("/health").json()
        r = c.post("/extract", json={"text": ENQUIRY, "mode": "rules_tagger"})
    assert health["tagger_loaded"] is False
    body = r.json()
    assert body["mode_requested"] == "rules_tagger" and body["mode_used"] == "rules"
    assert "not available" in body["note"]


def test_other_modes_do_not_use_the_tagger():
    with TestClient(api.create_app(load_model=False, crf_tagger=FakeTagger())) as c:
        body = c.post("/extract", json={"text": "We would like to see Zanzibar", "mode": "rules"}).json()
    assert body["profile"]["destination"] == []


# --- batch -------------------------------------------------------------------------------------


def test_batch_returns_one_profile_per_text_in_order(rules_client):
    texts = ["Two weeks in Kerala in March", "Malediven, 10 Nächte, 2 Pers.", "Nous partons au Népal en octobre"]
    r = rules_client.post("/extract/batch", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and body["mode_used"] == "rules"
    profiles = [api.ClientProfile.model_validate(p) for p in body["profiles"]]
    assert [p.destination for p in profiles] == [["Kerala"], ["Maldives"], ["Nepal"]]
    assert [p.language for p in profiles] == ["en", "de", "fr"]


def test_batch_with_the_tagger_marks_crf_values():
    with TestClient(api.create_app(load_model=False, crf_tagger=FakeTagger(), crf_tagger_v2=FakeTagger())) as c:
        assert c.get("/health").json()["tagger_v2_loaded"] is True
        body = c.post("/extract/batch", json={"texts": ["We would like to see Zanzibar"], "mode": "rules_tagger_v2"}).json()
    assert body["mode_used"] == "rules_tagger_v2"
    assert body["profiles"][0]["destination"] == ["Zanzibar"]


def test_batch_falls_back_to_rules_without_the_new_tagger():
    with TestClient(api.create_app(load_model=False, load_tagger=False)) as c:
        body = c.post("/extract/batch", json={"texts": [ENQUIRY], "mode": "rules_tagger_v2"}).json()
    assert body["mode_used"] == "rules" and "not available" in body["note"]


@pytest.mark.parametrize("payload, expected", [
    ({"texts": []}, "texts must have at least one text"),
    ({"texts": ["Kerala"] * 51}, "texts must have at most 50 texts"),
    ({"texts": ["Kerala", "x" * 20_001]}, "texts.1 must be at most 20000 characters"),
    ({"texts": "Kerala"}, "texts must be a list of texts"),
    ({"texts": ["Kerala", "   "]}, "texts text number 1 is empty (counting from 0)"),
])
def test_batch_rejects_bad_input_without_echoing_it(rules_client, payload, expected):
    r = rules_client.post("/extract/batch", json=payload)
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert expected in errors
    assert "Kerala" not in " ".join(errors) and "xxxx" not in " ".join(errors)


def test_batch_of_fifty_long_texts_is_accepted_by_the_size_limit(rules_client):
    r = rules_client.post("/extract/batch", json={"texts": ["Kerala in March. " * 1100] * 50})
    assert r.status_code == 200 and r.json()["count"] == 50
