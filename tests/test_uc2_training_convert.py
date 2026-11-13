"""Tests for the MASSIVE and Taskmaster-2 converters, using small made-up rows."""

import json
from collections import Counter

import pytest

from src.uc2_extractor.training.common import finish, trim
from src.uc2_extractor.training.convert_massive import convert_row, good_date, parse_annotation
from src.uc2_extractor.training.convert_taskmaster import classify, convert, convert_turn
from src.uc2_extractor.training.dedupe import ProtectedIndex
from src.uc2_extractor.training.labels import validate_example

# --- MASSIVE ------------------------------------------------------------------------------------


def row(annot, scenario="calendar", locale="de-DE"):
    return {"id": "7", "locale": locale, "partition": "train", "scenario": scenario, "annot_utt": annot}


def test_parse_annotation_gives_plain_text_and_offsets():
    text, slots = parse_annotation("buy a [transport_type : train] ticket for [date : friday]")
    assert text == "buy a train ticket for friday"
    assert [(s, text[a:b]) for s, a, b in slots] == [("transport_type", "train"), ("date", "friday")]


@pytest.mark.parametrize("annot, date", [
    ("add lunch on [date : march seventh]", "march seventh"),
    ("rendez-vous [date : vingt quatre mars]", "vingt quatre mars"),
    ("ein termin am [date : siebten märz]", "siebten märz"),
])
def test_date_slot_becomes_date(annot, date):
    example = convert_row(row(annot), "de", Counter())
    assert [(s["label"], s["text"]) for s in example["spans"]] == [("DATE", date)]
    validate_example(example)


def test_place_time_and_timeofday_are_ignored_not_labelled():
    example = convert_row(row("zug nach [place_name : hamburg] am [date : freitag] um [time : acht uhr]"), "de", Counter())
    assert [s["label"] for s in example["spans"]] == ["DATE"]
    assert [s["text"] for s in example["ignore"]] == ["hamburg", "acht uhr"]


@pytest.mark.parametrize("value", ["yesterday", "gestern", "la semaine dernière", "vierzehn uhr", "two hundred p. m.",
                                   "date", "un", "one thousand nine hundred and ninety"])
def test_past_clock_and_stray_values_are_not_dates(value):
    assert not good_date(value)


def test_date_that_is_not_clean_goes_to_ignore_and_row_is_dropped_if_nothing_is_left():
    assert convert_row(row("was war [date : gestern]"), "de", Counter()) is None


def test_other_scenarios_are_left_out():
    assert convert_row(row("play it [date : tomorrow]", scenario="alarm"), "en", Counter()) is None


def test_one_value_cannot_flood_the_data():
    seen = Counter({("en", "today"): 25})
    assert convert_row(row("add it [date : today]"), "en", seen) is None


# --- Taskmaster-2 ------------------------------------------------------------------------------


@pytest.mark.parametrize("name, value, expected", [
    ("hotel_search.location.hotel", "San Diego", "DESTINATION"),
    ("flight_search.destination1", "Rome", "DESTINATION"),
    ("flight_search.origin", "JFK", "ORIGIN"),
    ("hotel_search.check-in_date", "May 9th", "DATE"),
    ("flight_search.date.return", "April 7th", "DATE"),
    ("hotel_search.date_range", "June 16th to June 19th", "DATE"),
    ("hotel_search.date_range", "two nights", "DURATION"),
    ("hotel_search.num.guests", "two people", "PARTY"),
    ("hotel_search.num.guests", "just me", "PARTY"),
    ("hotel_search.num.guests", "two adults", "ADULTS"),
    ("hotel_search.num.guests", "me and my wife", "ADULTS"),
    ("hotel_search.star_rating", "four-star hotel", "ACCOMMODATION"),
    ("flight_search.seating_class", "business class", "FLIGHT"),
    ("flight_search.stops", "non-stop", "FLIGHT"),
])
def test_clean_values_are_kept(name, value, expected):
    assert classify(name, value)[0][0] == expected


@pytest.mark.parametrize("name, value", [
    ("hotel_search.num.guests", "two"),  # meaningless without the question before it
    ("hotel_search.num.guests", "a kid so I"),
    ("flight_search.num.pax", "one ticket"),
    ("flight_search.date.return", "in 4 days"),
    ("flight_search.date.return", "return in 5 days"),
    ("hotel_search.check-out_date", "leave Monday evening"),
    ("hotel_search.date_range", "three nights in May"),
    ("hotel_search.star_rating", "Class"),
    ("flight_search.stops", "layover"),
    ("flight_search.seating_class", "first"),
    ("hotel_search.price_range", "under $300 a night"),
    ("flight_search.price_range", "$700"),
    ("hotel_search.amenity", "swimming pool"),
    ("hotel_search.other_request", "good reviews"),
    ("hotel_search.sub_location.hotel", "French Quarter"),
])
def test_values_that_do_not_fit_are_ignored(name, value):
    assert classify(name, value) is None


def test_adults_and_children_in_one_span_are_split():
    parts = classify("hotel_search.num.guests", "Two adults and two children")
    assert [(p[0], p[3]) for p in parts] == [("ADULTS", 2), ("CHILDREN", 2)]


def test_values_are_canonical():
    assert classify("flight_search.seating_class", "coach")[0][3] == "economy class"
    assert classify("flight_search.stops", "nonstop")[0][3] == "direct flights"
    assert classify("hotel_search.date_range", "a week")[0][3] == "1 weeks"


def segment(text, part, *names):
    start = text.index(part)
    return {"start_index": start, "end_index": start + len(part), "text": part,
            "annotations": [{"name": n} for n in names]}


def test_turn_keeps_good_spans_and_ignores_the_rest():
    text = "A hotel in Boston under $200 a night, please."
    spans, ignore = convert_turn(text, [segment(text, "Boston", "hotel_search.location.hotel"),
                                        segment(text, "under $200 a night", "hotel_search.price_range")])
    assert [(s["label"], s["text"]) for s in spans] == [("DESTINATION", "Boston")]
    assert [p["text"] for p in ignore] == ["under $200 a night"]


def test_trailing_punctuation_is_trimmed_from_spans():
    text = "Fly to Rome."
    spans, _ = convert_turn(text, [segment(text, "Rome.", "flight_search.destination1")])
    assert spans[0]["text"] == "Rome"


def test_two_annotations_that_disagree_are_ignored():
    text = "Fly to Rome now"
    spans, ignore = convert_turn(text, [segment(text, "Rome", "flight_search.destination1", "flight_search.origin")])
    assert spans == [] and ignore[0]["text"] == "Rome"


def test_overlapping_labels_that_disagree_are_both_dropped():
    text = "Leaving Rome today"
    spans, ignore = convert_turn(text, [segment(text, "Rome today", "flight_search.origin"),
                                        segment(text, "today", "flight_search.date.depart_origin")])
    assert spans == []
    assert {p["text"] for p in ignore} == {"Rome today", "today"}


def test_whole_file_conversion_uses_user_turns_only(tmp_path):
    def utterance(i, speaker, text, segments=()):
        return {"index": i, "speaker": speaker, "text": text, "segments": list(segments)}

    text = "I need a flight to Lisbon on March 3rd"
    dialog = {"conversation_id": "dlg-1", "instruction_id": "x", "utterances": [
        utterance(0, "ASSISTANT", "Where to? Lisbon or Rome?",
                  [segment("Where to? Lisbon or Rome?", "Lisbon", "flight_search.destination1")]),
        utterance(1, "USER", text, [segment(text, "Lisbon", "flight_search.destination1"),
                                    segment(text, "March 3rd", "flight_search.date.depart_origin")]),
        utterance(2, "USER", "Yes that is perfect thank you so much"),
    ]}
    (tmp_path / "flights.json").write_text(json.dumps([dialog]), encoding="utf-8")
    (tmp_path / "hotels.json").write_text("[]", encoding="utf-8")
    examples = convert(tmp_path)
    assert len([e for e in examples if e["spans"]]) == 1
    assert examples[0]["source"] == "taskmaster2" and examples[0]["language"] == "en"
    for e in examples:
        validate_example(e)


# --- finishing steps ----------------------------------------------------------------------------


def test_trim_moves_edges_past_spaces_and_punctuation():
    text = ' "Rome". '
    a, b = trim(text, 0, len(text))
    assert text[a:b] == "Rome"


def test_finish_drops_invalid_too_close_and_repeated_examples():
    index = ProtectedIndex()
    index.add("We would really like to visit the old harbour market in the morning.")
    good = {"id": "1", "source": "t", "language": "en", "text": "Two weeks in Nepal", "spans": []}
    repeat = dict(good, id="2", text="two weeks in nepal!")
    close = dict(good, id="3", text="We would really like to visit the old harbour market in the morning.")
    broken = dict(good, id="4", spans=[{"start": 0, "end": 3, "label": "NOPE", "text": "Two"}])
    kept, dropped = finish([good, repeat, close, broken], index)
    assert [e["id"] for e in kept] == ["1"]
    assert dropped == {"exact_repeat": 1, "too_close_to_test_text:exact": 1, "failed_validation": 1}
