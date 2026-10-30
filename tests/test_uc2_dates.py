import pytest

from src.uc2_extractor.dates import parse_dates
from src.uc2_extractor.schema import ClientProfile, check_evidence


def s(pd):
    return pd.to_str() if pd else None


# --- ambiguous numeric dates are flagged, never guessed ----------------------


@pytest.mark.parametrize("text", ["Arriving 3/4, leaving 17/4?", "Ankunft am 3.4.", "le 3/4/2027"])
def test_ambiguous_numeric_date_is_not_used_as_start(text):
    r = parse_dates(text)
    assert r.dates.start is None or r.dates.start.to_str() != "--04-03"
    first = r.dates.ambiguous[0]
    assert first.text.startswith("3")
    readings = {x.to_str()[-5:] for x in first.readings}
    assert readings == {"04-03", "03-04"}
    assert any(f.kind == "ambiguous" and f.field == "dates" for f in r.flags)


def test_ambiguous_date_keeps_written_year():
    r = parse_dates("Start 3/4/2027")
    assert {x.to_str() for x in r.dates.ambiguous[0].readings} == {"2027-04-03", "2027-03-04"}


def test_two_digit_year():
    r = parse_dates("from 15/03/27")
    assert s(r.dates.start) == "2027-03-15"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("arriving 13/04", "--04-13"),  # 13 can only be a day
        ("arriving 04/13", "--04-13"),  # 13 can only be a day (month first)
        ("arriving 5/5", "--05-05"),  # same either way
        ("Anreise 15.03.2027", "2027-03-15"),
    ],
)
def test_unambiguous_numeric_dates(text, expected):
    r = parse_dates(text)
    assert s(r.dates.start) == expected
    assert r.dates.ambiguous == []


@pytest.mark.parametrize("text", ["a 4.5-star hotel", "budget 20.000 Euro", "2/3 adults", "up to 2.5k"])
def test_numbers_that_are_not_dates(text):
    r = parse_dates(text)
    assert r.dates.start is None and r.dates.ambiguous == []


# --- month names in three languages ------------------------------------------


@pytest.mark.parametrize(
    "text, start, end",
    [
        ("We want to travel from 3 to 17 March 2027.", "2027-03-03", "2027-03-17"),
        ("March 3-17", "--03-03", "--03-17"),
        ("between the 10th and the 24th of December", "--12-10", "--12-24"),
        ("Nous partirions du 3 au 17 mars.", "--03-03", "--03-17"),
        ("du 1er au 15 avril 2027", "2027-04-01", "2027-04-15"),
        ("Wir reisen vom 3. bis 17. März.", "--03-03", "--03-17"),
        ("vom 3. bis zum 17. Oktober 2027", "2027-10-03", "2027-10-17"),
        ("from 28 March to 4 April", "--03-28", "--04-04"),
        ("from 20 December to 5 January", "--12-20", "--01-05"),
        ("arrivée le 2 février, départ le 16 février", "--02-02", "--02-16"),
        ("Anreise am 2. Februar, Abreise am 16. Februar", "--02-02", "--02-16"),
    ],
)
def test_date_ranges(text, start, end):
    r = parse_dates(text)
    assert s(r.dates.start) == start
    assert s(r.dates.end) == end


@pytest.mark.parametrize(
    "text, start",
    [
        ("sometime in March", "--03"),
        ("early November 2027", "2027-11"),
        ("en décembre", "--12"),
        ("mi-mars", "--03"),
        ("im Oktober", "--10"),
        ("Anfang Mai", "--05"),
        ("Januar 2027", "2027-01"),
        ("on 3rd of March", "--03-03"),
        ("le 1er avril", "--04-01"),
    ],
)
def test_month_only_and_single_dates(text, start):
    r = parse_dates(text)
    assert s(r.dates.start) == start
    assert r.dates.end is None


def test_may_as_verb_is_not_a_date():
    r = parse_dates("We may travel with friends.")
    assert r.dates.start is None


def test_no_year_is_invented():
    r = parse_dates("from 3 to 17 March")
    assert r.dates.start.year is None and r.dates.end.year is None


# --- durations -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, value, unit",
    [
        ("two weeks in Kerala", 2, "weeks"),
        ("14 nights", 14, "nights"),
        ("a 10-day tour", 10, "days"),
        ("a fortnight", 2, "weeks"),
        ("deux semaines", 2, "weeks"),
        ("dix nuits", 10, "nights"),
        ("zwei Wochen", 2, "weeks"),
        ("eine 14-tägige Reise", 14, "days"),
        ("eine zweiwöchige Reise", 2, "weeks"),
        ("12 Nächte", 12, "nights"),
    ],
)
def test_durations(text, value, unit):
    d = parse_dates(text).dates.duration
    assert (d.value, d.unit) == (value, unit)


@pytest.mark.parametrize("text", ["twice a day", "USD 900 a night", "breakfast every day"])
def test_rates_and_frequencies_are_not_durations(text):
    assert parse_dates(text).dates.duration is None


def test_several_durations_without_total_are_flagged():
    r = parse_dates("3 nights in Delhi and 4 nights in Agra")
    assert r.dates.duration is None
    assert any("several trip lengths" in f.message for f in r.flags)


def test_total_duration_is_preferred():
    r = parse_dates("3 nights in Delhi, 4 nights in Agra, 12 nights in total")
    assert r.dates.duration.value == 12


# --- evidence ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Wir reisen vom 3. bis 17. März, also zwei Wochen.",
        "Arriving 3/4, about 10 nights.",
        "du 1er au 15 avril 2027 (quatorze nuits)",
    ],
)
def test_evidence_points_at_the_text(text):
    r = parse_dates(text)
    assert r.evidence
    profile = ClientProfile(dates=r.dates, evidence=r.evidence)
    assert check_evidence(profile, text) == []


def test_nothing_found_means_empty():
    r = parse_dates("We would love to see tigers.")
    assert r.dates.start is None and r.dates.duration is None
    assert r.evidence == [] and r.flags == []


# --- regressions found on the first gold-set run ---------------------------------------


def test_sentence_full_stop_is_not_part_of_slash_date():
    r = parse_dates("We land in Colombo on 5/6/2027. Thanks")
    assert r.dates.ambiguous[0].text == "5/6/2027"


def test_german_trailing_dot_stays_with_dotted_date():
    r = parse_dates("Abreise 02.01. Bitte")
    assert r.dates.ambiguous[0].text == "02.01."


def test_iso_date_followed_by_duration():
    r = parse_dates("ab dem 2027-01-12 10 Nächte")
    assert s(r.dates.start) == "2027-01-12"
    assert r.dates.duration.to_str() == "10 nights"


def test_duration_range_is_flagged_not_guessed():
    r = parse_dates("maybe 7-8 nights??")
    assert r.dates.duration is None
    assert any("range" in f.message for f in r.flags)


def test_whole_trip_length_wins_over_part_lengths():
    r = parse_dates("a 9-night trip: Mumbai (2 nights) then Goa (7 nights)")
    assert r.dates.duration.to_str() == "9 nights"
    r = parse_dates("un voyage de 9 nuits : Mumbai (2 nuits) puis Goa (7 nuits)")
    assert r.dates.duration.to_str() == "9 nights"


def test_alternative_months_are_not_a_range():
    r = parse_dates("Late October or early November, 10 nights.")
    assert r.dates.start is None and r.dates.end is None
    assert any("alternative" in f.message for f in r.flags)


def test_mid_month_with_hyphen():
    r = parse_dates("late January to mid-February")
    assert (s(r.dates.start), s(r.dates.end)) == ("--01", "--02")
