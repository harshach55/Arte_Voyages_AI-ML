import csv
import json

import pytest

from src.uc2_extractor.corrections import (
    REVIEW_STATUS,
    CorrectionError,
    count_corrections,
    save_correction,
    to_heldout_csv,
    values_to_profile,
)
from src.uc2_extractor.extract import extract_profile
from src.uc2_extractor.metrics import ALL_FIELDS, profile_to_values

TEXT = "My wife and I want 10 nights in Kerala from 3 to 13 March 2027, budget USD 9,000. Arriving 3/4?"


def test_values_round_trip_through_the_profile():
    extracted = extract_profile(TEXT)
    values = profile_to_values(extracted)
    rebuilt = values_to_profile(values, extracted)
    assert profile_to_values(rebuilt) == values


def test_editing_deleting_and_adding_values():
    extracted = extract_profile(TEXT)
    values = profile_to_values(extracted)
    values["destination"] = "Kerala | Goa"  # add a missing value
    values["budget_max"] = "9500"  # edit a value
    values["dates_ambiguous"] = ""  # delete a value
    values["interests"] = "beach"
    p = values_to_profile(values, extracted)
    assert p.destination == ["Kerala", "Goa"]
    assert p.budget.amount_max == 9500
    assert p.dates.ambiguous == []
    assert p.interests == ["beach"]


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("party_size", "two", "not a number"),
        ("start_date", "March", "not a date"),
        ("duration", "ten nights", "must look like"),
        ("language", "es", "language"),
        ("budget_min", "20000", "budget"),  # minimum above maximum
        ("dates_ambiguous", "5/6", "not found by the extraction"),
    ],
)
def test_invalid_corrections_are_refused_with_a_clear_message(field, value, message):
    extracted = extract_profile(TEXT)
    values = profile_to_values(extracted)
    values[field] = value
    with pytest.raises(CorrectionError, match=message):
        values_to_profile(values, extracted)


def test_unknown_field_is_refused():
    with pytest.raises(CorrectionError, match="unknown fields"):
        values_to_profile({"hotel": "x"})


def test_save_appends_one_line_per_correction(tmp_path):
    path = tmp_path / "corrections" / "corrections.jsonl"
    extracted = extract_profile(TEXT)
    values = profile_to_values(extracted)
    assert count_corrections(path) == 0
    assert save_correction(TEXT, extracted, values, path) == 1
    values["interests"] = "food and cooking"
    assert save_correction(TEXT, extracted, values, path) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    assert set(record) == {"timestamp", "language", "text", "extracted", "corrected"}
    assert record["text"] == TEXT and record["language"] == "en"
    assert record["extracted"]["interests"] == [] and record["corrected"]["interests"] == ["food and cooking"]
    assert record["timestamp"].endswith("+00:00")


def test_invalid_correction_is_not_saved(tmp_path):
    path = tmp_path / "corrections.jsonl"
    extracted = extract_profile(TEXT)
    values = profile_to_values(extracted)
    values["adults"] = "-1"
    with pytest.raises(CorrectionError):
        save_correction(TEXT, extracted, values, path)
    assert count_corrections(path) == 0


def test_convert_corrections_to_heldout_csv(tmp_path):
    path = tmp_path / "corrections.jsonl"
    for text, extra in [(TEXT, "beach"), ("Nous voulons 7 nuits aux Maldives.", "diving and snorkelling")]:
        extracted = extract_profile(text)
        values = profile_to_values(extracted)
        values["interests"] = extra
        save_correction(text, extracted, values, path)

    out = tmp_path / "as_labels"
    assert to_heldout_csv(path, out) == 2
    with open(out / "enquiries.csv", encoding="utf-8-sig", newline="") as f:
        enquiries = list(csv.DictReader(f))
    with open(out / "labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    assert [e["id"] for e in enquiries] == ["C0001", "C0002"]
    assert all(e["review_status"] == REVIEW_STATUS for e in enquiries)
    assert enquiries[1]["language"] == "fr"
    assert list(enquiries[0]) == ["id", "group_id", "language", "text", "review_status", "review_note",
                                  "second_annotator", "checked_by", "checked_on"]
    assert len(labels) == 2 * len(ALL_FIELDS)
    gold = {(r["id"], r["field"]): r["gold_value"] for r in labels}
    assert gold[("C0001", "interests")] == "beach"
    assert gold[("C0002", "interests")] == "diving and snorkelling"
    assert gold[("C0002", "destination")] == "Maldives"


def test_converted_files_work_with_the_evaluation(tmp_path):
    from src.uc2_extractor.evaluate import evaluate, load_gold

    path = tmp_path / "corrections.jsonl"
    extracted = extract_profile(TEXT)
    save_correction(TEXT, extracted, profile_to_values(extracted), path)
    to_heldout_csv(path, tmp_path / "out")
    enquiries, labels = load_gold(tmp_path / "out")
    field_table, *_ = evaluate(enquiries, labels)
    assert int(field_table.fn.sum()) == 0 and int(field_table.fp.sum()) == 0
