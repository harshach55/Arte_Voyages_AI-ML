"""Tests for the synthetic enquiry generator: labels match the text, and test text is kept out."""

import re
from collections import Counter

import pytest

from src.uc2_extractor.training.dedupe import ProtectedIndex
from src.uc2_extractor.training.labels import validate_example
from src.uc2_extractor.training.synth.generate import generate, make_example
from src.uc2_extractor.training.synth.parts import S, T, add_typos, assemble, tidy

NEW_LABELS = {"AGE", "BUDGET", "INTEREST", "DIETARY", "OCCASION", "SPECIAL_REQUEST"}


@pytest.fixture(scope="module")
def examples():
    kept, _ = generate(60, seed=123, index=ProtectedIndex())
    return kept


def test_every_example_is_valid(examples):
    assert len(examples) == 180
    for e in examples:
        validate_example(e)  # each span matches the text at its offsets and survives the BIO round trip
        assert all(e["text"][s["start"]:s["end"]] == s["text"] for s in e["spans"])


@pytest.mark.parametrize("language", ["en", "fr", "de"])
def test_each_language_has_the_fields_public_data_lacks(examples, language):
    labels = {s["label"] for e in examples if e["language"] == language for s in e["spans"]}
    assert NEW_LABELS <= labels


def test_all_three_styles_and_some_typos_appear(examples):
    styles = Counter(e["meta"]["style"] for e in examples)
    assert set(styles) == {"long", "medium", "short"}
    assert any(e["meta"]["typo_rate"] for e in examples)


def test_same_seed_gives_the_same_data():
    first, _ = generate(5, seed=99, index=ProtectedIndex())
    second, _ = generate(5, seed=99, index=ProtectedIndex())
    assert [e["text"] for e in first] == [e["text"] for e in second]


def test_text_close_to_a_protected_text_is_rejected():
    import random

    blocked = make_example("en", 0, random.Random("1-en"))["text"]
    index = ProtectedIndex()
    index.add(blocked)
    kept, dropped = generate(1, seed=1, index=index)
    assert blocked not in [e["text"] for e in kept]
    assert sum(n for reason, n in dropped.items() if reason.startswith("too_close")) >= 1


def test_no_sentence_is_repeated_inside_one_enquiry(examples):
    for e in examples:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", e["text"]) if len(s.split()) >= 6]
        assert len(sentences) == len(set(sentences)), e["id"]


def test_one_traveller_never_writes_we_in_english(examples):
    for e in examples:
        if e["language"] == "en" and e["meta"]["persona"] == "solo":
            assert not re.search(r"\bwe\b", e["text"], re.IGNORECASE), e["id"]


def test_typos_never_move_a_label():
    import random

    parts = [T("We would love "), S("INTEREST", "tiger safaris", "wildlife"), T(" in March.")]
    text, spans = assemble(add_typos(parts, 1.0, random.Random(3)))
    assert text[spans[0]["start"]:spans[0]["end"]] == spans[0]["text"] != "tiger safaris"


def test_tidy_drops_a_double_full_stop():
    text, spans = assemble(tidy([T("Budget "), S("BUDGET", "5.000 Euro p.P.", None), T(".")]))
    assert text == "Budget 5.000 Euro p.P." and spans[0]["text"] == "5.000 Euro p.P."
