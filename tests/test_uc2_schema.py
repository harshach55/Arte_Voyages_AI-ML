import json

import pytest
from pydantic import ValidationError

from src.uc2_extractor.schema import (
    BandConfig,
    Budget,
    ClientProfile,
    Evidence,
    Flag,
    PartialDate,
    TravelDates,
    Travellers,
    band_for_amount,
    check_evidence,
    load_band_config,
)


def test_empty_profile_is_valid_and_all_fields_empty():
    p = ClientProfile()
    assert p.language is None
    assert p.destination == []
    assert p.dates.start is None and p.dates.duration is None
    assert p.travellers.party_size is None
    assert p.budget.amount_min is None and p.budget.band is None


def test_full_profile_round_trips_through_json():
    p = ClientProfile(
        language="de",
        destination=["Sri Lanka"],
        dates=TravelDates(
            start=PartialDate(year=2027, month=3, day=3),
            end=PartialDate(year=2027, month=3, day=17),
            duration={"value": 14, "unit": "nights"},
        ),
        travellers=Travellers(party_size=4, adults=2, children=2, ages=[8, 11]),
        budget=Budget(amount_min=20000, amount_max=20000, currency="USD", band="10k_to_25k"),
        dietary=["vegetarian"],
        evidence=[Evidence(field="destination", text="Sri Lanka", start=10, end=19)],
        flags=[Flag(field="occasion", kind="missing", message="no occasion")],
    )
    again = ClientProfile.model_validate(json.loads(p.model_dump_json()))
    assert again == p


def test_unknown_top_level_field_is_rejected():
    with pytest.raises(ValidationError):
        ClientProfile(hotel_name="x")


def test_unsupported_language_is_rejected():
    with pytest.raises(ValidationError):
        ClientProfile(language="es")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"party_size": 0},
        {"ages": [-1]},
        {"ages": [130]},
        {"party_size": 4, "adults": 2, "children": 1},
        {"party_size": 1, "ages": [30, 32]},
    ],
)
def test_invalid_travellers_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        Travellers(**kwargs)


def test_partial_travellers_are_allowed():
    t = Travellers(adults=2)
    assert t.party_size is None and t.children is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"year": 2027, "month": 2, "day": 29},
        {"month": 4, "day": 31},
        {"month": 13},
        {"month": 0},
        {"year": 1990, "month": 1},
    ],
)
def test_impossible_dates_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        PartialDate(**kwargs)


def test_29_february_without_year_is_allowed():
    assert PartialDate(month=2, day=29).to_str() == "--02-29"


@pytest.mark.parametrize("text", ["2027-03-04", "--03-04", "2027-12", "--12"])
def test_partial_date_text_round_trip(text):
    assert PartialDate.from_str(text).to_str() == text


def test_end_before_start_is_rejected_when_years_known():
    with pytest.raises(ValidationError):
        TravelDates(
            start=PartialDate(year=2027, month=3, day=17),
            end=PartialDate(year=2027, month=3, day=3),
        )


def test_range_over_new_year_without_year_is_allowed():
    d = TravelDates(start=PartialDate(month=12, day=20), end=PartialDate(month=1, day=5))
    assert d.end.month == 1


def test_ambiguous_date_needs_two_readings():
    with pytest.raises(ValidationError):
        TravelDates(ambiguous=[{"text": "3/4", "readings": [{"month": 4, "day": 3}]}])


def test_budget_min_above_max_is_rejected():
    with pytest.raises(ValidationError):
        Budget(amount_min=30000, amount_max=20000, currency="USD")


def test_budget_band_must_exist_in_config():
    with pytest.raises(ValidationError):
        Budget(amount_min=5, currency="USD", band="cheap")


def test_budget_band_not_allowed_for_other_currency():
    with pytest.raises(ValidationError):
        Budget(amount_min=20000, currency="EUR", band="10k_to_25k")


def test_budget_band_not_allowed_per_person():
    with pytest.raises(ValidationError):
        Budget(amount_min=8000, currency="USD", basis="per_person", band="under_10k")


def test_currency_must_be_iso_code():
    with pytest.raises(ValidationError):
        Budget(currency="$")


def test_band_config_file_loads_and_is_marked_placeholder():
    config = load_band_config()
    assert config.currency == "USD"
    assert config.basis == "per_trip"
    assert config.placeholder is True
    assert [b.label for b in config.band] == ["under_10k", "10k_to_25k", "25k_to_50k", "over_50k"]


@pytest.mark.parametrize(
    "amount, label",
    [
        (0, "under_10k"),
        (9999.99, "under_10k"),
        (10000, "10k_to_25k"),
        (24999, "10k_to_25k"),
        (25000, "25k_to_50k"),
        (50000, "over_50k"),
        (2_000_000, "over_50k"),
    ],
)
def test_band_edges(amount, label):
    assert band_for_amount(amount) == label


def test_band_config_with_gap_is_rejected():
    with pytest.raises(ValidationError):
        BandConfig(
            currency="USD",
            basis="per_trip",
            band=[{"label": "a", "max": 10}, {"label": "b", "min": 20}],
        )


def test_evidence_span_must_match_text_length():
    with pytest.raises(ValidationError):
        Evidence(field="destination", text="India", start=0, end=3)


def test_evidence_field_must_be_known():
    with pytest.raises(ValidationError):
        Evidence(field="hotel", text="x", start=0, end=1)


def test_check_evidence_finds_wrong_offsets():
    text = "We want to visit India in March."
    good = ClientProfile(evidence=[Evidence(field="destination", text="India", start=17, end=22)])
    bad = ClientProfile(evidence=[Evidence(field="destination", text="India", start=16, end=21)])
    assert check_evidence(good, text) == []
    assert len(check_evidence(bad, text)) == 1
