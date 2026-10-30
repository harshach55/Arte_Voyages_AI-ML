import pytest

from src.uc2_extractor.extract import (
    _parse_amount,
    extract_budget,
    extract_flights,
    extract_profile,
    extract_travellers,
)
from src.uc2_extractor.schema import check_evidence

# --- nothing is invented ----------------------------------------------------------


def test_text_without_details_gives_empty_profile_and_missing_flags():
    p = extract_profile("Hello, please send me some ideas. Thank you!")
    assert p.destination == [] and p.accommodation == [] and p.interests == []
    assert p.dates.start is None and p.dates.duration is None
    assert p.travellers.party_size is None
    assert p.budget.amount_min is None and p.budget.band is None
    missing = {f.field for f in p.flags if f.kind == "missing"}
    assert {"destination", "dates", "travellers", "budget"} <= missing


def test_empty_text():
    p = extract_profile("")
    assert p.language is None and p.evidence == []


@pytest.mark.parametrize(
    "text",
    [
        "My wife and I want 2 weeks in Kerala, budget $15,000 per person, vegetarian, private guide.",
        "Nous sommes 4 personnes, Inde du Sud du 3 au 17 mars, budget 20 000 € au total, villa privée.",
        "Wir reisen mit zwei Kindern (8 und 11 Jahre alt) im Oktober nach Nepal, Budget 25.000 Euro.",
    ],
)
def test_every_evidence_span_matches_the_text(text):
    p = extract_profile(text)
    assert p.evidence
    assert check_evidence(p, text) == []


# --- lexicon fields ------------------------------------------------------------------


def test_negated_interest_is_dropped():
    p = extract_profile("We love temples but no shopping please.")
    assert p.interests == ["culture and heritage"]


@pytest.mark.parametrize("text", ["Pas de shopping, merci.", "Kein Shopping bitte."])
def test_negation_in_french_and_german(text):
    assert extract_profile(text).interests == []


def test_non_vegetarian_is_not_vegetarian():
    assert extract_profile("We are non-vegetarian.").dietary == []


def test_longer_match_wins():
    p = extract_profile("We would like an overwater villa in the Maldives.")
    assert p.accommodation == ["overwater villa"]


@pytest.mark.parametrize(
    "text, occasion",
    [
        ("pour notre anniversaire de mariage", ["anniversary"]),
        ("pour mon anniversaire", ["birthday"]),
        ("zu unserem Hochzeitstag", ["anniversary"]),
        ("für meinen 60. Geburtstag", ["birthday"]),
        ("our honeymoon", ["honeymoon"]),
    ],
)
def test_occasions(text, occasion):
    assert extract_profile(text).occasion == occasion


def test_destinations_are_reported_in_english_once():
    p = extract_profile("Wir möchten nach Indien. Indien ist unser Traum, dann Malediven.")
    assert p.destination == ["India", "Maldives"]


def test_word_inside_another_word_is_not_matched():
    assert extract_profile("Indian Ocean views").destination == []


# --- flights -------------------------------------------------------------------------


def test_departure_city_needs_a_flight_sentence():
    labels, _ = extract_flights("We are from London. We want to see tigers.")
    assert labels == []
    labels, _ = extract_flights("We would fly from London in business class.")
    assert labels == ["business class", "departure from London"]


@pytest.mark.parametrize(
    "text, label",
    [
        ("vols directs au départ de Genève", "departure from Geneva"),
        ("Flug ab Frankfurt", "departure from Frankfurt"),
        ("Our flights are already booked.", "flights not needed"),
        ("Die Flüge buchen wir selbst.", "flights not needed"),
    ],
)
def test_flight_labels(text, label):
    labels, _ = extract_flights(text)
    assert label in labels


# --- travellers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, party, adults, children, ages",
    [
        ("2 adults and 2 children aged 8 and 11", 4, 2, 2, [8, 11]),
        ("my husband and I", 2, 2, None, []),
        ("ma femme et moi avec nos deux enfants", 4, 2, 2, []),
        ("meine Frau und ich mit unserem Sohn (10 Jahre alt)", 3, 2, 1, [10]),
        ("my wife and I with our daughter", 3, 2, 1, []),
        ("deux adultes et une fille", 3, 2, 1, []),
        ("two sons and our daughter", None, None, 3, []),
        ("I am travelling solo", 1, 1, None, []),
        ("a family of five", 5, None, None, []),
        ("eine vierköpfige Familie", 4, None, None, []),
        ("nous sommes 6 personnes", 6, None, None, []),
        ("two adults, two sons and one daughter", 5, 2, 3, []),
        ("2 adults with a baby", 3, 2, 1, []),
        ("deux enfants de 6 et 9 ans", None, None, 2, [6, 9]),
    ],
)
def test_travellers(text, party, adults, children, ages):
    t, _, _ = extract_travellers(text)
    assert (t.party_size, t.adults, t.children, t.ages) == (party, adults, children, ages)


@pytest.mark.parametrize("text", ["elle fête ses 60 ans", "nos 25 ans de mariage", "depuis 20 ans"])
def test_french_years_that_are_not_ages(text):
    t, _, _ = extract_travellers(text)
    assert t.ages == []


def test_inconsistent_counts_are_flagged_not_forced():
    t, _, flags = extract_travellers("We are 5 people: 2 adults and 2 children.")
    assert t.party_size == 5 and t.adults is None and t.children is None
    assert flags and flags[0].kind == "ambiguous"


def test_we_are_followed_by_duration_is_not_a_count():
    t, _, _ = extract_travellers("wir werden zwei Wochen reisen")
    assert t.party_size is None


# --- budget --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, value",
    [("20,000", 20000), ("20.000", 20000), ("20 000", 20000), ("20'000", 20000),
     ("20k", 20000), ("2.5k", 2500), ("2,5", 2.5), ("15000", 15000), ("30 mille", 30000)],
)
def test_parse_amount(raw, value):
    assert _parse_amount(raw) == value


@pytest.mark.parametrize(
    "text, lo, hi, cur, band",
    [
        ("Our budget is USD 20,000 for the whole trip.", 20000, 20000, "USD", "10k_to_25k"),
        ("between US$20,000 and US$30,000 in total", 20000, 30000, "USD", None),  # spans two bands
        ("up to 45k USD in total", None, 45000, "USD", "25k_to_50k"),
        ("budget 25 000 € pour tout le monde", 25000, 25000, "EUR", None),  # no conversion
        ("Budget zwischen 20.000 und 30.000 Euro", 20000, 30000, "EUR", None),
        ("£12,000 total", 12000, 12000, "GBP", None),
    ],
)
def test_budget(text, lo, hi, cur, band):
    b, ev, _ = extract_budget(text)
    assert (b.amount_min, b.amount_max, b.currency, b.band) == (lo, hi, cur, band)
    assert ev


def test_per_person_budget_gets_no_band():
    b, _, flags = extract_budget("USD 9,000 per person")
    assert b.basis == "per_person" and b.band is None
    assert any("per person" in f.message for f in flags)


def test_bare_dollar_is_flagged_as_assumption():
    b, _, flags = extract_budget("about $40,000 in total")
    assert b.currency == "USD" and b.band == "25k_to_50k"
    assert any(f.kind == "assumption" and "US dollars" in f.message for f in flags)


def test_unknown_basis_is_flagged_as_assumption():
    _, _, flags = extract_budget("USD 20,000")
    assert any("whole trip" in f.message for f in flags)


def test_nightly_rate_is_not_the_budget():
    b, _, _ = extract_budget("hotels around USD 900 a night, total budget USD 30,000")
    assert b.amount_min == 30000


def test_budget_without_currency_is_kept_and_flagged():
    b, _, flags = extract_budget("Our budget is roughly 20,000.")
    assert b.amount_min == 20000 and b.currency is None and b.band is None
    assert any("currency" in f.message for f in flags)


def test_numbers_without_currency_or_budget_word_are_ignored():
    b, ev, _ = extract_budget("children aged 8 and 11, 14 nights")
    assert b.amount_min is None and ev == []
