import pytest

from src.uc2_extractor.extract import extract_profile
from src.uc2_extractor.metrics import ALL_FIELDS, Counts, compare_field, profile_to_values


def counts(gold, pred, field, lenient=False):
    c = compare_field(gold, pred, field, lenient).counts
    return (c.tp, c.fp, c.fn, c.extra_on_empty)


# --- scalar fields ---------------------------------------------------------------


def test_scalar_exact_match():
    assert counts("EUR", "eur", "currency") == (1, 0, 0, 0)


def test_scalar_both_empty_counts_nothing():
    assert counts("", "", "currency") == (0, 0, 0, 0)


def test_scalar_wrong_value_is_one_fp_and_one_fn():
    assert counts("EUR", "USD", "currency") == (0, 1, 1, 0)


def test_scalar_missed():
    assert counts("EUR", "", "currency") == (0, 0, 1, 0)


def test_scalar_invented_value_is_counted_as_extra_on_empty():
    assert counts("", "USD", "currency") == (0, 1, 0, 1)


def test_numbers_are_normalised():
    assert counts("30000", "30000.0", "budget_min") == (1, 0, 0, 0)


def test_numbers_get_no_lenient_credit():
    assert counts("4", "5", "party_size", lenient=True) == (0, 1, 1, 0)


@pytest.mark.parametrize(
    "gold, pred, ok",
    [
        ("--03", "--03-03", True),
        ("2027-03", "--03-15", True),
        ("--03-03", "--03-04", False),
        ("2027-03-03", "2028-03-03", False),
        ("--03", "--04", False),
    ],
)
def test_lenient_dates(gold, pred, ok):
    assert compare_field(gold, pred, "start_date", lenient=True).counts.tp == int(ok)
    assert compare_field(gold, pred, "start_date", lenient=False).counts.tp == int(gold == pred)


@pytest.mark.parametrize(
    "gold, pred, ok",
    [("14 nights", "14 days", True), ("2 weeks", "14 nights", True), ("2 weeks", "10 days", False)],
)
def test_lenient_durations(gold, pred, ok):
    assert compare_field(gold, pred, "duration", lenient=True).counts.tp == int(ok)


# --- list fields -------------------------------------------------------------------


def test_list_order_does_not_matter():
    assert counts("India | Nepal", "Nepal | India", "destination") == (2, 0, 0, 0)


def test_list_partial_match():
    r = compare_field("India | Nepal | Bhutan", "India | Delhi", "destination")
    assert (r.counts.tp, r.counts.fp, r.counts.fn) == (1, 1, 2)
    assert r.extra == ["delhi"]
    assert sorted(r.missed) == ["bhutan", "nepal"]


def test_list_duplicates_are_counted_as_multiset():
    assert counts("28 | 28", "28", "ages") == (1, 0, 1, 0)
    assert counts("28", "28 | 28", "ages") == (1, 1, 0, 0)


def test_list_lenient_substring():
    assert counts("Ladakh | Pangong Lake", "Ladakh | Pangong", "destination") == (1, 1, 1, 0)
    assert counts("Ladakh | Pangong Lake", "Ladakh | Pangong", "destination", lenient=True) == (2, 0, 0, 0)


def test_list_invented_when_gold_empty():
    assert counts("", "beach", "interests") == (0, 1, 0, 1)


# --- rates ---------------------------------------------------------------------------


def test_precision_recall_f1():
    c = Counts(tp=6, fp=2, fn=4)
    assert c.precision == pytest.approx(0.75)
    assert c.recall == pytest.approx(0.6)
    assert c.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_rates_are_undefined_without_data():
    c = Counts()
    assert c.precision is None and c.recall is None and c.f1 is None


def test_zero_true_positives_gives_f1_zero():
    assert Counts(tp=0, fp=3, fn=2).f1 == 0.0


# --- profile flattening --------------------------------------------------------------


def test_profile_to_values_has_every_field_and_gold_format():
    p = extract_profile(
        "My wife and I want to visit Sri Lanka from 3 to 17 March 2027 with our two kids aged 8 and 11. "
        "Budget USD 30,000 in total."
    )
    v = profile_to_values(p)
    assert set(v) == set(ALL_FIELDS)
    assert v["destination"] == "Sri Lanka"
    assert v["start_date"] == "2027-03-03" and v["end_date"] == "2027-03-17"
    assert v["party_size"] == "4" and v["ages"] == "8 | 11"
    assert v["budget_min"] == "30000" and v["budget_band"] == "25k_to_50k"


def test_empty_profile_flattens_to_empty_strings():
    v = profile_to_values(extract_profile("Hello"))
    assert all(v[f] == "" for f in ALL_FIELDS if f != "language")
