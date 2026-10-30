"""Travel date and duration parsing for English, French and German enquiries.

Principles:
- never guess: a numeric date such as 3/4 where both numbers could be the
  month is kept as ambiguous with both readings, and is not used as a start
  or end date
- never invent a year: if the year is not written next to the date it stays
  empty
- if several different trip lengths are mentioned and none is marked as the
  total, the duration stays empty and is flagged
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from .lexicon import MONTHS, NUMBER_WORDS, alternation
from .schema import AmbiguousDate, Duration, Evidence, Flag, PartialDate, TravelDates

_I = re.IGNORECASE
_MONTH_ALT = alternation(MONTHS)


def _month(name: str) -> str:
    return rf"(?P<{name}>{_MONTH_ALT})\.?(?!\w)"


def _day(name: str) -> str:
    # 3, 3rd, 1er, 3. (German ordinal); not part of a longer number
    return rf"(?<![\d.,/])(?P<{name}>[0-3]?\d)(?:st|nd|rd|th|er|\.)?(?!\d)"


_YEAR = r"(?:,?\s*(?P<y>20\d{2})(?!\d))?"
_TO = (
    r"\s*(?:-|–|—|to|till|until|through|au|à|jusqu'au|bis zum|bis|and|et|und)\s*"
    r"(?:the\s+|le\s+|zum\s+|den\s+)?"
)
_OF = r"(?:\s+of)?\s+"
_NOT_COUNT = (
    r"(?!\s*(?:nights?|days?|weeks?|adults?|people|persons?|guests|pax|children|kids|"
    r"personnes|personen|enfants|kinder|erwachsene|stars?))"
)

PATTERNS = [
    ("iso", re.compile(r"(?<!\d)(?P<y>20\d{2})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})(?!\d)")),
    ("range_dm", re.compile(_day("d1") + _TO + _day("d2") + _OF + _month("m") + _YEAR, _I)),
    ("range_md", re.compile(_month("m") + r"\s+" + _day("d1") + _TO + _day("d2") + _YEAR, _I)),
    ("single_dm", re.compile(_day("d") + _OF + _month("m") + _YEAR, _I)),
    ("single_md", re.compile(_month("m") + r"\s+" + _day("d") + _NOT_COUNT + _YEAR, _I)),
    (
        "numeric",
        re.compile(
            r"(?<![\d.,/])(?P<a>\d{1,2})/(?P<b>\d{1,2})(?:/(?P<y>\d{4}|\d{2}))?"
            r"(?![\d/])(?!\s*-?\s*(?:stars?|star|étoiles?|sterne|%|k\b))" + _NOT_COUNT,
            _I,
        ),
    ),
    (
        # German style 15.03. or 15.03.2027 (a trailing dot belongs to the date)
        "numeric",
        re.compile(
            r"(?<![\d.,/])(?P<a>\d{1,2})\.(?P<b>\d{1,2})(?:\.(?P<y>\d{4}|\d{2})|\.)?"
            r"(?![\d/])(?!\s*-?\s*(?:stars?|star|étoiles?|sterne|%|k\b))" + _NOT_COUNT,
            _I,
        ),
    ),
    (
        "month_only",
        re.compile(
            r"(?:\b(?:in|during|early|late|mid|end of|beginning of|start of|en|au mois de|"
            r"début|fin|courant|im|anfang|ende|mitte)\s+|\bmid?-)" + _month("m") + _YEAR,
            _I,
        ),
    ),
    ("month_year", re.compile(_month("m") + r"\s+(?P<y>20\d{2})(?!\d)", _I)),
]

# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

UNITS = {
    "night": "nights", "nights": "nights", "nuit": "nights", "nuits": "nights",
    "nacht": "nights", "nächte": "nights", "nächten": "nights",
    "day": "days", "days": "days", "jour": "days", "jours": "days",
    "tag": "days", "tage": "days", "tagen": "days",
    "week": "weeks", "weeks": "weeks", "semaine": "weeks", "semaines": "weeks",
    "woche": "weeks", "wochen": "weeks",
}
_DUR_NUMBERS = {**NUMBER_WORDS, "a": 1, "an": 1}
DURATION_RE = re.compile(
    rf"(?<!\w)(?P<n>\d{{1,3}}|{alternation(_DUR_NUMBERS)})(?:\s*-\s*|\s+)(?P<u>{alternation(UNITS)})(?!\w)",
    _I,
)
# German compounds: 14-tägige Reise, zweiwöchige Reise
DURATION_DE_RE = re.compile(
    rf"(?<!\w)(?P<n>\d{{1,3}}|{alternation(NUMBER_WORDS)})\s*-?\s*(?P<u>tägig|wöchig)\w*",
    _I,
)
FORTNIGHT_RE = re.compile(r"(?<!\w)(?:a\s+)?fortnight(?!\w)", _I)
# "twice a day", "$500 a night": a rate or frequency, not a trip length
_NOT_TRIP_LENGTH = re.compile(r"\b(?:once|twice|per|times|every|each|par|pro|mal)\s*$", _I)
_RATE_BEFORE = re.compile(r"[\d$€£]\s*$")
# "7-8 nights", "10 or 12 days": a range, so no single trip length
_RANGE_BEFORE = re.compile(r"\d\s*(?:-|–|to|or|à|ou|bis|oder)\s*$", _I)
_TOTAL_HINT = re.compile(r"\b(?:total|in all|insgesamt|au total|en tout|gesamt)\b", _I)
# "9-night trip", "voyage de 9 nuits", "Reise mit 9 Nächten": length of the whole trip
_TRIP_AFTER = re.compile(r"^\s*(?:trip|tour|journey|holiday|vacation|itinerary)\b", _I)
_TRIP_BEFORE = re.compile(
    r"\b(?:voyage|s[ée]jour|circuit|reise|rundreise|urlaub)\s+(?:de|d'|mit|von)\s*(?:environ\s+|etwa\s+)?$", _I
)


@dataclass
class _Mention:
    kind: str  # "range", "date", "month", "ambiguous"
    start: int
    end: int
    text: str
    first: Optional[PartialDate] = None
    last: Optional[PartialDate] = None
    readings: list[PartialDate] = field(default_factory=list)


@dataclass
class DateResult:
    dates: TravelDates
    evidence: list[Evidence]
    flags: list[Flag]


def _year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    y = int(text)
    return y + 2000 if y < 100 else y


def _pd(day, month, year=None) -> Optional[PartialDate]:
    try:
        return PartialDate(year=year, month=month, day=day)
    except ValidationError:
        return None


def _to_mention(kind: str, m: re.Match) -> Optional[_Mention]:
    g = m.groupdict()
    year = _year(g.get("y"))
    base = dict(start=m.start(), end=m.end(), text=m.group())

    if kind == "iso":
        d = _pd(int(g["d"]), int(g["mo"]), year)
        return _Mention("date", first=d, **base) if d else None

    if kind in ("range_dm", "range_md"):
        month = MONTHS[g["m"].lower()]
        first = _pd(int(g["d1"]), month, year)
        last = _pd(int(g["d2"]), month, year)
        if first and last and first.day < last.day:
            return _Mention("range", first=first, last=last, **base)
        return None

    if kind in ("single_dm", "single_md"):
        d = _pd(int(g["d"]), MONTHS[g["m"].lower()], year)
        return _Mention("date", first=d, **base) if d else None

    if kind == "numeric":
        a, b = int(g["a"]), int(g["b"])
        if a <= 12 and b <= 12 and a != b:
            readings = [r for r in (_pd(a, b, year), _pd(b, a, year)) if r]
            if len(readings) == 2:
                return _Mention("ambiguous", readings=readings, **base)
            return None
        if b <= 12:  # day first (13/04) or same number (5/5)
            d = _pd(a, b, year)
        elif a <= 12:  # month first (04/13)
            d = _pd(b, a, year)
        else:
            d = None
        return _Mention("date", first=d, **base) if d else None

    if kind in ("month_only", "month_year"):
        d = PartialDate(year=year, month=MONTHS[g["m"].lower()])
        return _Mention("month", first=d, **base)

    return None


def _find_mentions(text: str) -> list[_Mention]:
    taken: list[tuple[int, int]] = []
    mentions = []
    for kind, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            mention = _to_mention(kind, m)
            if mention:
                taken.append((m.start(), m.end()))
                mentions.append(mention)
    return sorted(mentions, key=lambda x: x.start)


def _find_durations(text: str) -> tuple[list[tuple[Duration, re.Match]], list[re.Match]]:
    """Return (durations, ranges). Ranges such as '7-8 nights' are kept apart."""
    found = []
    ranges = []
    for m in DURATION_RE.finditer(text):
        before = text[max(0, m.start() - 8):m.start()]
        if _NOT_TRIP_LENGTH.search(before):
            continue
        n = m.group("n").lower()
        if n in ("a", "an") and _RATE_BEFORE.search(before):
            continue
        if _RANGE_BEFORE.search(before):
            ranges.append(m)
            continue
        value = int(n) if n.isdigit() else _DUR_NUMBERS[n]
        found.append((Duration(value=value, unit=UNITS[m.group("u").lower()]), m))
    for m in DURATION_DE_RE.finditer(text):
        n = m.group("n").lower()
        value = int(n) if n.isdigit() else NUMBER_WORDS[n]
        unit = "days" if m.group("u").lower() == "tägig" else "weeks"
        found.append((Duration(value=value, unit=unit), m))
    for m in FORTNIGHT_RE.finditer(text):
        found.append((Duration(value=2, unit="weeks"), m))
    return sorted(found, key=lambda x: x[1].start()), ranges


def _is_whole_trip(text: str, m: re.Match) -> bool:
    if _TOTAL_HINT.search(text[max(0, m.start() - 20):m.end() + 20]):
        return True
    return bool(_TRIP_AFTER.search(text[m.end():m.end() + 15])
                or _TRIP_BEFORE.search(text[max(0, m.start() - 25):m.start()]))


def _evidence(m_start: int, m_end: int, text: str) -> Evidence:
    return Evidence(field="dates", text=text[m_start:m_end], start=m_start, end=m_end)


def parse_dates(text: str) -> DateResult:
    mentions = _find_mentions(text)
    evidence: list[Evidence] = []
    flags: list[Flag] = []

    ranges = [m for m in mentions if m.kind == "range"]
    days = [m for m in mentions if m.kind == "date"]
    months = [m for m in mentions if m.kind == "month"]
    ambiguous = [m for m in mentions if m.kind == "ambiguous"]

    used: list[_Mention] = []
    start = end = None
    if ranges:
        start, end = ranges[0].first, ranges[0].last
        used = [ranges[0]]
    elif days:
        used = days[:2]
    elif months:
        used = months[:2]
    if len(used) == 2 and re.fullmatch(r"\s*(?:or|ou|oder)\s*", text[used[0].end:used[1].start], _I):
        # "late October or early November": a choice, not a range
        flags.append(Flag(field="dates", kind="ambiguous",
                          message=f"alternative dates: '{text[used[0].start:used[1].end]}'"))
        used = []
    if not ranges and used:
        start = used[0].first
        end = used[1].first if len(used) > 1 else None

    for m in ambiguous:
        readings = ", ".join(r.to_str() for r in m.readings)
        flags.append(
            Flag(
                field="dates",
                kind="ambiguous",
                message=f"'{m.text}' can be read as {readings}; please confirm",
            )
        )

    # duration
    duration = None
    durations, duration_ranges = _find_durations(text)
    for rm in duration_ranges:
        start_of_range = max(0, rm.start() - 8)
        flags.append(Flag(field="dates", kind="ambiguous",
                          message=f"trip length given as a range: '{text[start_of_range:rm.end()].strip()}'"))
    distinct = {(d.value, d.unit) for d, _ in durations}
    if len(distinct) == 1:
        duration, dm = durations[0]
        evidence.append(_evidence(dm.start(), dm.end(), text))
    elif len(distinct) > 1:
        totals = [(d, dm) for d, dm in durations if _is_whole_trip(text, dm)]
        if len({(d.value, d.unit) for d, _ in totals}) == 1:
            duration, dm = totals[0]
            evidence.append(_evidence(dm.start(), dm.end(), text))
        else:
            listed = ", ".join(dm.group() for _, dm in durations)
            flags.append(
                Flag(
                    field="dates",
                    kind="ambiguous",
                    message=f"several trip lengths mentioned ({listed}); total not clear",
                )
            )

    try:
        dates = TravelDates(
            start=start,
            end=end,
            duration=duration,
            ambiguous=[AmbiguousDate(text=m.text, readings=m.readings) for m in ambiguous],
        )
    except ValidationError:
        # e.g. an end date before the start date: keep the start, flag the rest
        dates = TravelDates(start=start, duration=duration)
        flags.append(Flag(field="dates", kind="ambiguous", message="end date is before start date"))
        used = used[:1]

    for m in used + ambiguous:
        evidence.append(_evidence(m.start, m.end, text))
    evidence.sort(key=lambda e: e.start)
    return DateResult(dates=dates, evidence=evidence, flags=flags)
