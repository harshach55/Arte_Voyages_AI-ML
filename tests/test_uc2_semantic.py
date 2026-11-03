"""Tests for the optional model layer.

Most tests use a small fake embedder so they run without the model files. The
last group uses the real model and is skipped if it has not been downloaded.
"""

import zlib

import numpy as np
import pytest

from src.uc2_extractor import semantic
from src.uc2_extractor.extract import extract_profile
from src.uc2_extractor.schema import check_evidence

# Fake embedder: each concept is one dimension. Any other text gets its own
# pseudo-random direction (seeded from the text), so unrelated texts score
# close to zero against each other, as with a real model.
CONCEPTS = {
    0: ("wild animals", "birds", "oiseaux", "vögel", "tiere"),
    1: ("spa", "massage", "soins", "wellness"),
    2: ("palace", "palais", "palast"),
    3: ("own driver", "chauffeur", "fahrer"),
}
N_CONCEPTS = len(CONCEPTS)
DIM = N_CONCEPTS + 256


def fake_embed(texts):
    out = np.zeros((len(texts), DIM))
    for i, t in enumerate(texts):
        low = t.lower()
        for dim, words in CONCEPTS.items():
            if any(w in low for w in words):
                out[i, dim] = 1.0
        if not out[i].any():
            rng = np.random.default_rng(zlib.crc32(low.encode("utf-8")))
            out[i, N_CONCEPTS:] = rng.standard_normal(DIM - N_CONCEPTS)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def clear_banks():
    semantic._BANKS.clear()
    yield
    semantic._BANKS.clear()


# --- splitting -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "He is mad about birds, and I love old palaces. Budget £15,000!",
        "Nous rêvons de soins au spa (massages) et d'un chauffeur – merci",
        "Wir möchten Vögel beobachten und einen eigenen Fahrer; danke",
    ],
)
def test_pieces_keep_exact_offsets(text):
    pieces = semantic.split_pieces(text)
    assert pieces
    for p in pieces:
        assert text[p.start:p.end] == p.text


def test_long_clauses_are_also_cut_into_windows():
    text = "one two three four five six seven eight nine ten eleven twelve"
    pieces = semantic.split_pieces(text, window=6, step=3)
    assert pieces[0].text == text
    assert any(p.text == "one two three four five six" for p in pieces)


# --- matching logic ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["He is mad about birds", "Il est passionné d'oiseaux", "Er liebt es, Vögel zu sehen"],
)
def test_model_fills_an_empty_field(text):
    p = extract_profile(text, model=fake_embed, threshold=0.5)
    assert "wildlife" in p.interests
    prov = [x for x in p.provenance if x.field == "interests" and x.value == "wildlife"]
    assert prov and prov[0].source == "model" and 0 < prov[0].confidence <= 1
    model_ev = [e for e in p.evidence if e.source == "model"]
    assert model_ev and check_evidence(p, text) == []


@pytest.mark.parametrize(
    "text",
    ["We do not want to see any birds", "Pas d'oiseaux, merci", "Keine Vögel bitte"],
)
def test_negated_pieces_are_skipped(text):
    p = extract_profile(text, model=fake_embed, threshold=0.5)
    assert "wildlife" not in p.interests


def test_model_does_not_touch_a_field_the_rules_filled():
    # "temples" is found by the rules, so interests is not empty and the model stays out
    p = extract_profile("We love temples and birds", model=fake_embed, threshold=0.5)
    assert p.interests == ["culture and heritage"]
    assert all(x.source == "rule" for x in p.provenance if x.field == "interests")


def test_model_can_fill_other_empty_fields():
    p = extract_profile("We dream of an old palace hotel? No: a stay in a palais, with our own driver",
                        model=fake_embed, threshold=0.5)
    assert "heritage or palace hotel" in p.accommodation  # found by the rules ("palais")
    assert "private driver" in p.special_requests


def test_below_threshold_nothing_is_added():
    p = extract_profile("He is mad about birds", model=fake_embed, threshold=1.01)
    assert p.interests == []


def test_rules_only_is_the_default():
    p = extract_profile("He is mad about birds")
    assert p.interests == []
    assert all(x.source == "rule" for x in p.provenance)


def test_unrelated_text_adds_nothing():
    p = extract_profile("Please send the invoice to my office.", model=fake_embed, threshold=0.5)
    assert all(x.source == "rule" for x in p.provenance)


# --- the real model (skipped if not downloaded) -------------------------------------------------

real_model = pytest.mark.skipif(
    not (semantic.MODEL_DIR / semantic.ONNX_FILE).exists(), reason="model files not downloaded"
)


@real_model
def test_real_embedder_gives_unit_vectors():
    v = semantic.default_embedder()(["a test sentence", "eine Testsatz"])
    assert v.shape == (2, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


@real_model
@pytest.mark.parametrize(
    "text",
    ["My husband is crazy about birdlife", "Mon mari adore regarder les oiseaux", "Mein Mann liebt Vogelbeobachtung in der Natur"],
)
def test_real_model_ranks_wildlife_first(text):
    # checks the ranking, not a fixed cut-off: wildlife must be the best interest label
    matches = semantic.find_matches(text, semantic.default_embedder(), ["interests"], threshold=0.0)
    assert matches and matches[0].label == "wildlife"


@real_model
def test_windows_of_a_negated_clause_are_skipped():
    pieces = semantic.split_pieces("We are not interested in birds or animals at all, thank you")
    assert all(p.negated for p in pieces if "birds" in p.text)


@real_model
@pytest.mark.parametrize(
    "text",
    ["We are not interested in birds or animals at all", "Nous ne voulons pas voir d'animaux",
     "Wir wollen keine Tiere beobachten"],
)
def test_real_model_respects_negation(text):
    p = extract_profile(text, model=True)
    assert "wildlife" not in p.interests
