"""Tests for the trained tagger: features, split, metrics, offsets and filling empty fields."""

import json
from collections import Counter

import pytest

from src.uc2_extractor.extract import extract_profile
from src.uc2_extractor.schema import ClientProfile, Evidence, Provenance, check_evidence
from src.uc2_extractor.tagger import (
    CrfTagger,
    TaggedSpan,
    add_tagger_values,
    find_spans,
    spans_from_tags,
    tagger_profile,
    token_features,
    usable_spans,
    word_shape,
)
from src.uc2_extractor.text import tokenize
from src.uc2_extractor.training import train_crf
from src.uc2_extractor.training.compare_tagger import covered_fields
from src.uc2_extractor.training.labels import bio_to_spans, spans_to_bio

ALL_ENGLISH = {"DESTINATION", "ORIGIN", "DATE", "DURATION", "PARTY", "ADULTS", "CHILDREN", "ACCOMMODATION", "FLIGHT", "BUDGET"}


class FakeTagger:
    """Tags the phrases it is given, so tests do not depend on a trained model."""

    def __init__(self, phrases, confidence=0.9, allowed=None):
        self.phrases = phrases  # {"text": "LABEL"}
        self.confidence = confidence
        self.allowed = {"en": ALL_ENGLISH, "fr": {"DATE"}, "de": {"DATE"}} if allowed is None else allowed

    def labels_for(self, language):
        return self.allowed.get(language, {"DATE"})

    def predict(self, words, language):
        tags = ["O"] * len(words)
        low = [w.lower() for w in words]
        for phrase, label in self.phrases.items():
            part = [t.text.lower() for t in tokenize(phrase)]
            for i in range(len(low) - len(part) + 1):
                if low[i:i + len(part)] == part:
                    tags[i:i + len(part)] = [f"B-{label}"] + [f"I-{label}"] * (len(part) - 1)
        return tags, [self.confidence] * len(words)


def sources(profile, field):
    return [p for p in profile.provenance if p.field == field and p.source == "crf"]


# --- features ---------------------------------------------------------------------------------


def test_word_shape():
    assert word_shape("Delhi") == "Xx"
    assert word_shape("BA123") == "Xd"
    assert word_shape("20,000") == "d,d"


def test_features_have_everything_asked_for():
    f = token_features(["Fly", "to", "Paris", "in", "2027"], "fr")
    assert f[2]["w"] == "paris" and f[2]["lang"] == "fr" and f[2]["shape"] == "Xx"
    assert f[2]["pre3"] == "par" and f[2]["suf3"] == "ris"
    assert f[2]["w-1"] == "to" and f[2]["w+1"] == "in" and f[2]["w-2"] == "fly" and f[2]["w+2"] == "2027"
    assert f[4].get("digit") == 1.0 and "digit" not in f[2]


def test_features_mark_the_edges_of_the_sentence():
    f = token_features(["Goa"], None)
    assert f[0]["w-1"] == "<s>" and f[0]["w+1"] == "</s>" and f[0]["lang"] == "unknown"


# --- split and training data -------------------------------------------------------------------


def tm2(dialogue, turn):
    return {"id": f"tm2-hotels-{dialogue}-{turn}", "source": "taskmaster2", "language": "en", "meta": {"domain": "hotels"}}


def test_all_turns_of_a_dialogue_stay_on_one_side_of_the_split():
    for n in range(200):
        assert len({train_crf.is_dev(tm2(f"dlg-{n}", turn)) for turn in range(6)}) == 1


def test_split_takes_about_the_right_share_of_dialogues():
    share = sum(train_crf.is_dev(tm2(f"dlg-{n}", 1)) for n in range(2000)) / 2000
    assert 0.11 < share < 0.19


@pytest.mark.parametrize("language", ["en", "fr", "de"])
def test_massive_uses_its_own_partitions_so_translations_stay_together(language):
    row = {"id": f"massive-{language}-7", "source": "massive", "language": language}
    assert train_crf.is_dev({**row, "meta": {"partition": "train"}}) is False
    assert train_crf.is_dev({**row, "meta": {"partition": "dev"}}) is True
    assert train_crf.is_dev({**row, "meta": {"partition": "test"}}) is True


def test_ignored_text_is_left_out_of_training():
    text = "Hotel in Boston under $200 a night in May"
    a, b = text.index("under"), text.index(" in May")
    example = {"language": "en", "text": text,
               "spans": [{"start": 9, "end": 15, "label": "DESTINATION", "text": "Boston"},
                         {"start": len(text) - 3, "end": len(text), "label": "DATE", "text": "May"}],
               "ignore": [{"start": a, "end": b, "text": text[a:b]}]}
    pieces = train_crf.to_sequences(example)
    assert [p[1] for p in pieces] == [["O", "O", "B-DESTINATION"], ["O", "B-DATE"]]
    assert pieces[1][0][0]["w-1"] == "night"  # context is kept even though the words are not trained on


def test_label_stays_off_in_a_language_with_too_little_data():
    rows = [{"language": "en", "spans": [{"label": "DATE"}]}] * 20 + [{"language": "fr", "spans": [{"label": "DATE"}]}] * 19
    assert train_crf.labels_by_language(rows) == {"en": ["DATE"], "fr": [], "de": []}


# --- metrics -----------------------------------------------------------------------------------


def span(start, end, label):
    return {"start": start, "end": end, "label": label}


def test_span_counts_needs_exact_edges_and_label():
    gold = [span(0, 5, "DATE"), span(10, 15, "DESTINATION")]
    predicted = [span(0, 5, "DATE"), span(10, 14, "DESTINATION"), span(20, 22, "PARTY")]
    counts = train_crf.span_counts(gold, predicted)
    assert counts["DATE"] == [1, 0, 0]
    assert counts["DESTINATION"] == [0, 1, 1]
    assert counts["PARTY"] == [0, 1, 0]


def test_prediction_on_ignored_text_is_not_a_false_positive():
    counts = train_crf.span_counts([], [span(3, 8, "DESTINATION")], [{"start": 2, "end": 10}])
    assert not counts


def test_rates_by_hand():
    r = train_crf.rates(6, 2, 4)
    assert (r["precision"], r["recall"], r["f1"]) == (0.75, 0.6, 0.667)
    assert r["gold_spans"] == 10


def test_rates_with_nothing_found():
    assert train_crf.rates(0, 0, 0)["f1"] == 0.0


def test_score_uses_only_labels_of_that_language():
    example = {"language": "fr", "text": "Rome en mars", "spans": [span(8, 12, "DATE")], "ignore": []}
    tagger = FakeTagger({"Rome": "DESTINATION", "mars": "DATE"})
    report = train_crf.score(tagger, [example])
    assert report["fr"]["ALL"]["f1"] == 1.0 and "DESTINATION" not in report["fr"]


# --- offsets -----------------------------------------------------------------------------------


@pytest.mark.parametrize("text, tags", [
    ("Nous partons à Goa, début mars !", ["O", "O", "O", "B-DESTINATION", "O", "B-DATE", "I-DATE", "O"]),
    ("vom 3.4. bis 10.4.", ["O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE"]),
    ("Kerala Goa", ["B-DESTINATION", "B-DESTINATION"]),
    ("a I-tag first", ["O", "I-DATE", "O"]),
    ("change of label", ["B-DATE", "I-DESTINATION", "O"]),
])
def test_spans_match_the_text_and_agree_with_bio_to_spans(text, tags):
    tokens = tokenize(text)
    tags = (tags + ["O"] * len(tokens))[:len(tokens)]
    found = spans_from_tags(text, tokens, tags, [0.9] * len(tokens))
    assert all(text[s.start:s.end] == s.text for s in found)
    assert [(s.start, s.end, s.label) for s in found] == [(s["start"], s["end"], s["label"]) for s in bio_to_spans(text, tags)]


def test_span_confidence_is_its_weakest_token():
    text = "late May trip"
    found = spans_from_tags(text, tokenize(text), ["B-DATE", "I-DATE", "O"], [0.9, 0.6, 0.99])
    assert found[0].confidence == 0.6


# --- filling empty fields ----------------------------------------------------------------------


TEXT = "We would like to visit Zanzibar Bay from Porto with coach seats."
TAGS = {"Zanzibar Bay": "DESTINATION", "Porto": "ORIGIN", "coach seats": "FLIGHT"}


def test_tagger_fills_fields_the_rules_left_empty():
    rules = extract_profile(TEXT)
    assert rules.destination == [] and rules.flights == []
    p = extract_profile(TEXT, tagger=FakeTagger(TAGS))
    assert p.destination == ["Zanzibar Bay"]
    assert set(p.flights) == {"economy class", "departure from Porto"}
    assert [x.source for x in sources(p, "destination")] == ["crf"]
    assert all(x.confidence == 0.9 for x in sources(p, "flights"))
    assert all(e.source == "crf" for e in p.evidence if e.field in ("destination", "flights"))
    assert check_evidence(p, TEXT) == []
    ClientProfile.model_validate(p.model_dump())


def test_rule_values_are_never_replaced():
    text = "Two weeks in Kerala for 2 adults"
    fake = FakeTagger({"Two weeks": "DURATION", "Goa": "DESTINATION", "Kerala": "DESTINATION", "2 adults": "PARTY"})
    assert extract_profile(text, tagger=fake).model_dump() == extract_profile(text).model_dump()


def test_a_missing_destination_flag_goes_once_the_tagger_fills_it():
    assert any(f.field == "destination" for f in extract_profile(TEXT).flags)
    assert not any(f.field == "destination" for f in extract_profile(TEXT, tagger=FakeTagger(TAGS)).flags)


def test_low_confidence_spans_are_dropped():
    assert usable_spans(TEXT, "en", FakeTagger(TAGS, confidence=0.49)) == []
    assert len(usable_spans(TEXT, "en", FakeTagger(TAGS, confidence=0.5))) == 3


def test_negated_spans_are_dropped():
    text = "Please no Zanzibar Bay this time."
    assert usable_spans(text, "en", FakeTagger({"Zanzibar Bay": "DESTINATION"})) == []


def test_only_dates_count_in_french_and_german():
    text = "Nous aimerions aller à Zanzibar Bay."
    p = ClientProfile(language="fr")
    add_tagger_values(p, text, FakeTagger({"Zanzibar Bay": "DESTINATION"}))
    assert p.destination == []


def test_with_no_language_only_shared_labels_count():
    p = ClientProfile()
    add_tagger_values(p, "Zanzibar Bay", FakeTagger({"Zanzibar Bay": "DESTINATION"}))
    assert p.destination == []


def test_french_date_goes_through_the_date_parser():
    text = "Nous partirons un jour, peut-être du 3 au 12 mars."
    p = ClientProfile(language="fr")
    add_tagger_values(p, text, FakeTagger({"3 au 12 mars": "DATE"}))
    assert (p.dates.start.month, p.dates.start.day, p.dates.end.day) == (3, 3, 12)
    assert sources(p, "dates")[0].value == "start and end"


def test_a_second_date_closes_the_range():
    text = "Leaving May 9th and back May 16th."
    p = ClientProfile(language="en")
    add_tagger_values(p, text, FakeTagger({"May 9th": "DATE", "May 16th": "DATE"}))
    assert (p.dates.start.day, p.dates.end.day) == (9, 16)


def test_ambiguous_date_is_flagged_never_guessed():
    p = ClientProfile(language="en")
    add_tagger_values(p, "We leave on 3/4 next year.", FakeTagger({"3/4": "DATE"}))
    assert p.dates.start is None and [a.text for a in p.dates.ambiguous] == ["3/4"]
    assert any(f.kind == "ambiguous" for f in p.flags)


def test_unreadable_date_is_flagged_and_left_empty():
    p = ClientProfile(language="en")
    add_tagger_values(p, "Sometime next spring perhaps", FakeTagger({"next spring": "DATE"}))
    assert p.dates.start is None and not sources(p, "dates")
    assert any("next spring" in f.message and f.kind == "ambiguous" for f in p.flags)


def test_duration_goes_through_the_date_parser():
    p = ClientProfile(language="en")
    add_tagger_values(p, "Could be about ten nights overall", FakeTagger({"ten nights": "DURATION"}))
    assert p.dates.duration.to_str() == "10 nights"


@pytest.mark.parametrize("phrase, label, field, value", [
    ("two people", "PARTY", "party_size", 2),
    ("just me", "PARTY", "party_size", 1),
    ("me and my wife", "ADULTS", "adults", 2),
    ("3 adults", "ADULTS", "adults", 3),
    ("2 kids", "CHILDREN", "children", 2),
])
def test_counts_go_through_the_number_code(phrase, label, field, value):
    p = ClientProfile(language="en")
    add_tagger_values(p, f"It is for {phrase} this year", FakeTagger({phrase: label}))
    assert getattr(p.travellers, field) == value


def test_party_size_follows_from_adults_and_children():
    p = ClientProfile(language="en")
    add_tagger_values(p, "3 adults and 2 kids", FakeTagger({"3 adults": "ADULTS", "2 kids": "CHILDREN"}))
    assert p.travellers.party_size == 5


def test_counts_that_disagree_are_not_added():
    p = ClientProfile(language="en")
    p.travellers.party_size = 4
    add_tagger_values(p, "3 adults and 2 kids", FakeTagger({"3 adults": "ADULTS", "2 kids": "CHILDREN"}))
    assert (p.travellers.adults, p.travellers.children) == (3, None)


def test_budget_goes_through_the_money_code():
    text = "Spending roughly CHF 12,000 on this"
    p = ClientProfile(language="en")
    add_tagger_values(p, text, FakeTagger({"CHF 12,000": "BUDGET"}))
    assert (p.budget.amount_max, p.budget.currency) == (12000, "CHF")
    assert sources(p, "budget")[0].confidence == 0.9


def test_accommodation_without_a_label_adds_nothing():
    p = ClientProfile(language="en")
    add_tagger_values(p, "a four-star hotel", FakeTagger({"four-star hotel": "ACCOMMODATION"}))
    assert p.accommodation == []


def test_accommodation_is_normalised_to_its_label():
    p = ClientProfile(language="en")
    add_tagger_values(p, "a five-star hotel", FakeTagger({"five-star hotel": "ACCOMMODATION"}))
    assert p.accommodation == ["5-star"]


def test_tagger_profile_has_no_rule_values():
    p = tagger_profile("We want Kerala with vegetarian food", FakeTagger({"Kerala": "DESTINATION"}))
    assert p.destination == ["Kerala"] and p.dietary == []


def test_covered_fields():
    assert covered_fields(["DATE"]) == ["start_date", "end_date", "dates_ambiguous"]
    assert "flights" in covered_fields(["ORIGIN"]) and "ages" not in covered_fields(ALL_ENGLISH)


# --- schema ------------------------------------------------------------------------------------


def test_schema_accepts_crf_and_needs_its_confidence():
    assert Evidence(field="dates", text="May", start=0, end=3, source="crf").source == "crf"
    assert Provenance(field="dates", value="start", source="crf", confidence=0.7).confidence == 0.7
    with pytest.raises(ValueError):
        Provenance(field="dates", value="start", source="crf")


# --- a real, very small model ------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    sklearn_crfsuite = pytest.importorskip("sklearn_crfsuite")
    path = tmp_path_factory.mktemp("crf") / "small.crfsuite"
    rows = [("fly to rome tomorrow", [(7, 11, "DESTINATION")]), ("fly to paris now", [(7, 12, "DESTINATION")]),
            ("a trip to oslo please", [(10, 14, "DESTINATION")]), ("thanks very much", [])] * 5
    x, y = [], []
    for text, spans in rows:
        words, tags = spans_to_bio(text, [{"start": a, "end": b, "label": c, "text": text[a:b]} for a, b, c in spans])
        x.append(token_features(words, "en"))
        y.append(tags)
    sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.01, c2=0.01, max_iterations=30, model_filename=str(path)).fit(x, y)
    return path


def test_real_model_finds_spans_with_offsets_and_probabilities(small_model):
    tagger = CrfTagger(small_model, {"en": ["DESTINATION"]})
    text = "fly to rome tomorrow"
    found = find_spans(text, "en", tagger)
    assert [(s.label, s.text) for s in found] == [("DESTINATION", "rome")]
    assert text[found[0].start:found[0].end] == "rome" and 0.5 < found[0].confidence <= 1


def test_model_info_is_read_from_the_file_beside_it(small_model):
    small_model.with_suffix(".json").write_text(json.dumps({"labels_by_language": {"en": ["DATE"]}}), encoding="utf-8")
    assert CrfTagger(small_model).labels_for("en") == {"DATE"}


def test_missing_model_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CrfTagger(tmp_path / "nothing.crfsuite")


def test_span_type_is_plain():
    assert TaggedSpan("DATE", 0, 3, "May", 0.8).label == "DATE"


# --- the second tagger: new labels ---------------------------------------------------------------


class LabelFake(FakeTagger):
    """A fake tagger that also has a span-to-label classifier."""

    def __init__(self, phrases, labels, probability=0.8):
        super().__init__(phrases, allowed={"en": ALL_ENGLISH | {"AGE", "INTEREST", "DIETARY", "OCCASION", "SPECIAL_REQUEST"}})
        self.labels, self.probability = labels, probability

    def label_for(self, label, text):
        return self.labels.get(text), self.probability


def test_classifier_turns_unusual_wording_into_a_label():
    p = ClientProfile(language="en")
    add_tagger_values(p, "We long for big cats in the wild", LabelFake({"big cats in the wild": "INTEREST"},
                                                                        {"big cats in the wild": "wildlife"}))
    assert p.interests == ["wildlife"]
    assert sources(p, "interests")[0].confidence == 0.8  # the lower of span confidence (0.9) and label probability


def test_unsure_classifier_adds_nothing():
    p = ClientProfile(language="en")
    add_tagger_values(p, "We long for big cats in the wild", LabelFake({"big cats in the wild": "INTEREST"},
                                                                        {"big cats in the wild": "wildlife"}, 0.4))
    assert p.interests == []


def test_word_lists_come_before_the_classifier():
    p = ClientProfile(language="en")
    add_tagger_values(p, "We want tiger safaris", LabelFake({"tiger safaris": "INTEREST"}, {"tiger safaris": "beach"}))
    assert p.interests == ["wildlife"]


@pytest.mark.parametrize("label, phrase, field, value", [
    ("DIETARY", "no meat at all", "dietary", "vegetarian"),
    ("OCCASION", "our first trip as a married couple", "occasion", "honeymoon"),
    ("SPECIAL_REQUEST", "someone else to do the driving", "special_requests", "private driver"),
])
def test_other_label_fields_are_filled_the_same_way(label, phrase, field, value):
    p = ClientProfile(language="en")
    add_tagger_values(p, f"Note: {phrase}.", LabelFake({phrase: label}, {phrase: value}))
    assert getattr(p, field) == [value]


def test_ages_come_from_age_spans():
    p = ClientProfile(language="en")
    add_tagger_values(p, "2 adults and 2 kids, they are 7 and 10", LabelFake(
        {"2 adults": "ADULTS", "2 kids": "CHILDREN", "7 and 10": "AGE"}, {}))
    assert p.travellers.ages == [7, 10] and p.travellers.party_size == 4


def test_more_ages_than_travellers_are_not_added():
    p = ClientProfile(language="en")
    p.travellers.party_size = 1
    add_tagger_values(p, "ages 7 and 10", LabelFake({"7 and 10": "AGE"}, {}))
    assert p.travellers.ages == []


def test_synthetic_split_is_fixed_and_keeps_test_out_of_training():
    rows = [{"id": f"synth-en-{i:06d}", "source": "synthetic"} for i in range(1000)]
    parts = Counter(train_crf.split_of(r) for r in rows)
    assert parts == Counter(train_crf.split_of(r) for r in rows)
    assert 60 < parts["test"] < 140 and 60 < parts["dev"] < 140


def test_v2_name_picks_the_second_tagger(monkeypatch):
    from src.uc2_extractor import tagger

    monkeypatch.setattr(tagger, "default_tagger_v2", lambda: "second")
    assert tagger.resolve("v2") == "second"


# --- field-selective mode -------------------------------------------------------------------------


def test_field_selective_tagger_keeps_only_allowed_labels():
    from src.uc2_extractor.tagger import FieldSelectiveTagger

    base = LabelFake({"Zanzibar Bay": "DESTINATION", "big cats": "INTEREST"}, {"big cats": "wildlife"})
    p = ClientProfile(language="en")
    add_tagger_values(p, "Zanzibar Bay and big cats", FieldSelectiveTagger(base, ["INTEREST"]))
    assert p.destination == [] and p.interests == ["wildlife"]


def test_with_no_allowed_label_the_profile_equals_rules_only():
    from src.uc2_extractor.tagger import FieldSelectiveTagger

    text = "We would like to visit Zanzibar Bay from Porto with coach seats."
    selective = FieldSelectiveTagger(FakeTagger(TAGS), [])
    assert extract_profile(text, tagger=selective).model_dump() == extract_profile(text).model_dump()


def test_field_choice_needs_strictly_better_gold_scores():
    from src.uc2_extractor.metrics import Counts
    from src.uc2_extractor.training import select_fields as S

    def counts(tp, fp, fn):
        table = {(f, lang): Counts() for unit in S.UNITS.values() for f in unit[0] for lang in ("en", "fr", "de")}
        table[("interests", "en")] = Counts(tp=tp, fp=fp, fn=fn)
        return table

    assert S.select(counts(5, 1, 2), counts(6, 1, 1))["interests"]["chosen"] is True   # better
    assert S.select(counts(5, 1, 2), counts(5, 1, 2))["interests"]["chosen"] is False  # a tie is not chosen
    assert S.select(counts(5, 1, 2), counts(5, 2, 2))["interests"]["chosen"] is False  # worse


def test_field_choice_refuses_a_held_out_set(capsys):
    from src.uc2_extractor.training import select_fields as S

    assert S.main(["--gold-dir", "data/uc2_inquiries/heldout"]) == 2
    assert "gold set only" in capsys.readouterr().out
