import json

import pytest

from src.uc2_extractor.training.dedupe import ProtectedIndex, build_index, normalise
from src.uc2_extractor.training.labels import (
    IGNORE_TAG,
    LABELS,
    TAGS,
    ExampleError,
    bio_to_spans,
    spans_to_bio,
    validate_example,
)


def example(text, spans, **extra):
    return {"id": "x1", "source": "test", "language": "en", "text": text, "spans": spans, **extra}


def span(text, part, label, start=None):
    s = text.index(part) if start is None else start
    return {"start": s, "end": s + len(part), "label": label, "text": part}


# --- scheme and BIO ---------------------------------------------------------------------------


def test_tags_are_o_plus_begin_and_inside_for_every_label():
    assert TAGS[0] == "O"
    assert len(TAGS) == 1 + 2 * len(LABELS)
    assert len(set(TAGS)) == len(TAGS)


@pytest.mark.parametrize(
    "text, part, label",
    [
        ("Ten nights in Sri Lanka please", "Sri Lanka", "DESTINATION"),
        ("Nous aimerions aller à l'île de Ceylan", "l'île de Ceylan", "DESTINATION"),
        ("Budget etwa 15.000 € pro Person", "etwa 15.000 € pro Person", "BUDGET"),
        ("about ~$12k in total", "~$12k", "BUDGET"),
    ],
)
def test_bio_round_trip_keeps_exact_offsets(text, part, label):
    spans = [span(text, part, label)]
    tokens, tags = spans_to_bio(text, spans)
    assert len(tokens) == len(tags)
    assert tags.count(f"B-{label}") == 1
    assert bio_to_spans(text, tags) == spans


def test_two_spans_next_to_each_other_stay_apart():
    text = "2 adults 3 kids"
    spans = [span(text, "2 adults", "ADULTS"), span(text, "3 kids", "CHILDREN")]
    _, tags = spans_to_bio(text, spans)
    assert tags == ["B-ADULTS", "I-ADULTS", "B-CHILDREN", "I-CHILDREN"]
    assert bio_to_spans(text, tags) == spans


def test_stray_inside_tag_starts_a_new_span():
    assert bio_to_spans("two weeks", ["I-DURATION", "I-DURATION"])[0]["text"] == "two weeks"


# --- validation --------------------------------------------------------------------------------


def test_good_example_passes():
    text = "Two weeks in Nepal"
    validate_example(example(text, [span(text, "Two weeks", "DURATION"), span(text, "Nepal", "DESTINATION")]))


@pytest.mark.parametrize(
    "spans, message",
    [
        ([{"start": 0, "end": 3, "label": "DATE", "text": "Tw"}], "does not match"),
        ([{"start": 0, "end": 99, "label": "DATE", "text": "x"}], "outside the text"),
        ([{"start": 0, "end": 3, "label": "HOTEL", "text": "Two"}], "unknown label"),
        ([{"start": 3, "end": 4, "label": "DATE", "text": " "}], "spaces at its edge"),
        ([{"start": 0, "end": 5, "label": "DATE", "text": "Two w"}], "token boundaries"),
        ([{"start": 0, "end": 2, "label": "DATE", "text": "Tw"}], "no whole token"),
        ([{"start": 0, "end": 9, "label": "DATE", "text": "Two weeks"},
          {"start": 4, "end": 9, "label": "DURATION", "text": "weeks"}], "overlaps"),
    ],
)
def test_bad_spans_are_refused(spans, message):
    with pytest.raises(ExampleError, match=message):
        validate_example(example("Two weeks in Nepal", spans))


def test_ignored_text_gets_the_ignore_tag_and_no_label():
    text = "A room in Nepal under $300 a night"
    part = {"start": text.index("under"), "end": len(text), "text": "under $300 a night"}
    spans = [span(text, "Nepal", "DESTINATION")]
    validate_example(example(text, spans, ignore=[part]))
    tokens, tags = spans_to_bio(text, spans, [part])
    assert tags[tokens.index("under"):] == [IGNORE_TAG] * len(tags[tokens.index("under"):])
    assert bio_to_spans(text, tags) == spans


def test_ignored_text_may_not_overlap_a_label():
    text = "Two weeks in Nepal"
    bad = {"start": 10, "end": 18, "text": "in Nepal"}
    with pytest.raises(ExampleError, match="overlaps a labelled span"):
        validate_example(example(text, [span(text, "Nepal", "DESTINATION")], ignore=[bad]))


def test_ignored_text_must_match_the_enquiry():
    with pytest.raises(ExampleError, match="does not match"):
        validate_example(example("Two weeks", [], ignore=[{"start": 0, "end": 3, "text": "One"}]))


def test_missing_key_and_bad_language_are_refused():
    with pytest.raises(ExampleError, match="missing key"):
        validate_example({"id": "x", "text": "a"})
    bad = example("Nepal", [])
    bad["language"] = "es"
    with pytest.raises(ExampleError, match="unknown language"):
        validate_example(bad)


# --- staying apart from test data --------------------------------------------------------------

PROTECTED = ("The orchard picnic near the old lighthouse was booked for our cousins last spring, "
             "and everyone asked about the blue ferry timetable.")


@pytest.fixture
def index():
    idx = ProtectedIndex()
    idx.add(PROTECTED)
    idx.add("Maldives, 5 nights, 2 ppl, overwater")
    return idx


def test_normalise_ignores_case_accents_and_punctuation():
    assert normalise("  Hôtel, à-Négombo!! ") == "hotel a negombo"


def test_exact_copy_is_caught_even_with_other_punctuation(index):
    assert index.check(PROTECTED.upper().replace(",", " ;")) == "exact"


def test_shared_run_of_six_words_is_caught(index):
    assert index.check("Sadly the old lighthouse was booked for our friends in June") == "shared_run"


def test_short_note_variant_is_caught(index):
    assert index.check("maldives 5 nights 2 ppl") in ("near_text", "near_chars")


def test_typo_variant_is_caught(index):
    typo = PROTECTED.replace("orchard", "orchrd").replace("lighthouse", "lighthuose").replace("cousins", "cuosins")
    assert index.check(typo) is not None


def test_sentence_inside_a_longer_text_is_caught():
    idx = ProtectedIndex()
    idx.add("Please arrange a quiet walk along the river with a local baker.")
    candidate = ("Good morning. We have many plans this year and our dates are open. "
                 "Please arrange one quiet walk along the river with some local baker. Thank you.")
    assert idx.check(candidate) is not None


def test_different_text_passes(index):
    assert index.check("We hope to see tigers in Nepal for two weeks next autumn with our son") is None


def test_index_holds_no_readable_text(index):
    stored = repr(index.exact) + repr(index.runs) + repr(index.docs) + repr(index.sentences)
    assert "lighthouse" not in stored and "orchard" not in stored


def test_build_index_reads_every_protected_set(tmp_path):
    for name, text in (("gold", "first enquiry here"), ("heldout", "second enquiry here"),
                       ("heldout2", "third enquiry here")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "enquiries.csv").write_text(f"id,text\n1,{text}\n", encoding="utf-8-sig")
    (tmp_path / "corrections").mkdir()
    line = json.dumps({"text": "fourth enquiry here"}) + "\n"
    (tmp_path / "corrections" / "corrections.jsonl").write_text(line, encoding="utf-8")
    index, counts = build_index(tmp_path)
    assert counts == {"gold": 1, "heldout": 1, "heldout2": 1, "corrections": 1}
    assert index.check("FOURTH enquiry here!") == "exact"


def test_missing_held_out_two_is_not_an_error(tmp_path):
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "enquiries.csv").write_text("id,text\n1,hello there friend\n", encoding="utf-8-sig")
    _, counts = build_index(tmp_path)
    assert counts == {"gold": 1}
