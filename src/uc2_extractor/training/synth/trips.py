"""Sample a language-neutral trip: who travels, where, when, for how much, and what they want.

A language module then writes the trip as an enquiry. Labels such as "wildlife" match the
fixed label lists in lexicon.py. Interests and places are grouped so that a trip hangs
together: a Maldives trip leans to beaches and diving, a Himalayan one to trekking.

The place list is geography, not test data: names that appear only in a test set and not
in the original place list are left out on purpose.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Optional

# region -> (country or regional names, towns and sites)
PLACE_GROUPS = {
    "north_india": (["India", "Rajasthan", "Madhya Pradesh", "Uttarakhand"],
                    ["Delhi", "Agra", "Jaipur", "Udaipur", "Jodhpur", "Jaisalmer", "Bundi", "Orchha", "Varanasi",
                     "Amritsar", "Khajuraho", "Ranthambore", "Bandhavgarh", "Kanha", "Rishikesh", "Shimla"]),
    "south_india": (["India", "Kerala", "Tamil Nadu", "Karnataka", "Goa"],
                    ["Kochi", "Munnar", "Wayanad", "Alleppey", "Madurai", "Pondicherry", "Mysuru", "Hampi", "Coorg",
                     "Chennai", "Mumbai", "Bengaluru"]),
    "himalaya": (["Nepal", "Bhutan", "Ladakh", "Sikkim", "Himachal Pradesh"],
                 ["Kathmandu", "Pokhara", "Chitwan", "Bardia", "Lumbini", "Paro", "Thimphu", "Punakha", "Bumthang",
                  "Leh", "Gangtok", "Darjeeling"]),
    "sri_lanka": (["Sri Lanka"],
                  ["Colombo", "Kandy", "Galle", "Sigiriya", "Ella", "Yala", "Trincomalee", "Nuwara Eliya",
                   "Anuradhapura", "Arugam Bay"]),
    "maldives": (["Maldives"], ["Baa Atoll", "Ari Atoll", "Malé"]),
}
GROUP_INTERESTS = {
    "north_india": ["culture and heritage", "wildlife", "food and cooking", "luxury train", "festivals", "shopping",
                    "photography", "spirituality"],
    "south_india": ["food and cooking", "wellness and yoga", "beach", "culture and heritage", "tea plantations",
                    "wildlife", "shopping"],
    "himalaya": ["trekking and hiking", "spirituality", "photography", "culture and heritage", "wildlife",
                 "wellness and yoga"],
    "sri_lanka": ["wildlife", "tea plantations", "culture and heritage", "beach", "food and cooking",
                  "diving and snorkelling"],
    "maldives": ["beach", "diving and snorkelling", "wellness and yoga", "golf"],
}
GROUP_ACCOMMODATION = {
    "north_india": ["heritage or palace hotel", "boutique hotel", "5-star", "suite", "tented camp", "safari lodge"],
    "south_india": ["houseboat", "boutique hotel", "villa", "beachfront", "eco-lodge", "5-star", "private pool"],
    "himalaya": ["boutique hotel", "eco-lodge", "suite", "tented camp", "5-star"],
    "sri_lanka": ["boutique hotel", "villa", "beachfront", "safari lodge", "private pool", "eco-lodge"],
    "maldives": ["overwater villa", "private pool", "beachfront", "villa", "suite"],
}
ORIGINS = ["London", "Manchester", "Edinburgh", "Dublin", "Paris", "Lyon", "Nice", "Geneva", "Zurich", "Basel",
           "Frankfurt", "Munich", "Berlin", "Hamburg", "Vienna", "Brussels", "Amsterdam", "Luxembourg",
           "New York", "Toronto", "Dubai"]
FLIGHT_LABELS = ["business class", "first class", "premium economy", "economy class", "direct flights",
                 "domestic flights"]
DIETARY_LABELS = ["vegetarian", "vegan", "gluten-free", "lactose-free", "nut allergy", "shellfish allergy",
                  "halal", "kosher", "jain", "diabetic"]
COUPLE_OCCASIONS = ["honeymoon", "anniversary", "birthday", "retirement", "proposal", "babymoon",
                    "Christmas or New Year"]
FAMILY_OCCASIONS = ["birthday", "anniversary", "graduation", "family reunion", "Christmas or New Year", "retirement"]
SOLO_OCCASIONS = ["birthday", "retirement", "graduation"]
GENERAL_SPECIAL = ["private guide", "private driver", "airport transfers", "butler service", "late checkout",
                   "helicopter", "short drives only", "accessibility needs"]
FAMILY_SPECIAL = ["childcare", "baby cot"]
# style -> field -> chance that the enquiry mentions it
PRESENCE = {
    "long": dict(dest=.95, date=.8, dur=.8, party=.9, budget=.7, accommodation=.75, flights=.5, interests=.85,
                 dietary=.35, occasion=.4, special=.5),
    "medium": dict(dest=.95, date=.7, dur=.7, party=.75, budget=.5, accommodation=.4, flights=.25, interests=.5,
                   dietary=.2, occasion=.25, special=.25),
    "short": dict(dest=.9, date=.5, dur=.6, party=.6, budget=.4, accommodation=.25, flights=.15, interests=.3,
                  dietary=.12, occasion=.12, special=.12),
}
CURRENCY_WEIGHTS = {"en": {"USD": 4, "GBP": 4, "EUR": 2, "CHF": 1, "INR": 1},
                    "fr": {"EUR": 7, "CHF": 2, "USD": 2, "GBP": 1, "INR": 1},
                    "de": {"EUR": 7, "CHF": 3, "USD": 2, "GBP": 1, "INR": 1}}


@dataclass
class DateInfo:
    kind: str  # month, part_month, range_month, range_cross, single, numeric_ambiguous, numeric_clear,
    #            month_range, month_choice
    month: int = 1
    month2: int = 1
    day: int = 1
    day2: int = 1
    year: Optional[int] = None
    part: str = ""  # early, mid or late


@dataclass
class Budget:
    low: Optional[int]
    high: Optional[int]
    currency: str
    basis: Optional[str]  # per_trip, per_person or None
    shape: str  # about, exact, up_to, at_least, range


@dataclass
class Trip:
    language: str
    style: str
    persona: str
    group: str
    places: list[str] = field(default_factory=list)
    adults: Optional[int] = None
    children: Optional[int] = None
    total: Optional[int] = None
    ages: list[int] = field(default_factory=list)
    date: Optional[DateInfo] = None
    duration: Optional[tuple[int, str]] = None
    budget: Optional[Budget] = None
    accommodation: list[str] = field(default_factory=list)
    flights: list[str] = field(default_factory=list)
    origin: Optional[str] = None
    no_flights: bool = False
    interests: list[str] = field(default_factory=list)
    dietary: list[str] = field(default_factory=list)
    occasion: Optional[str] = None
    special: list[str] = field(default_factory=list)


def _maybe(rng: random.Random, chance: float) -> bool:
    return rng.random() < chance


def _pick(rng: random.Random, options: list, low: int, high: int) -> list:
    return rng.sample(options, min(len(options), rng.randint(low, high)))


def _sample_people(trip: Trip, rng: random.Random) -> None:
    p = trip.persona
    if p == "couple":
        trip.adults, trip.total = 2, 2
        if _maybe(rng, .12):
            trip.ages = [rng.randint(28, 74), rng.randint(28, 74)]
    elif p == "solo":
        trip.adults, trip.total = 1, 1
        if _maybe(rng, .25):
            trip.ages = [rng.randint(24, 78)]
    elif p == "family":
        trip.children = rng.randint(1, 4)
        trip.adults = 2 if trip.children == 1 else rng.choice((1, 2, 2, 2))
        trip.total = trip.adults + trip.children
        trip.ages = [rng.randint(1, 17) for _ in range(trip.children)]
    elif p == "friends":
        trip.adults = trip.total = rng.choice((3, 4, 4, 5, 6))
    elif p == "two_couples":
        trip.adults = trip.total = rng.choice((4, 4, 6))
    elif p == "multigen":
        trip.adults = rng.randint(3, 6)
        trip.children = rng.randint(2, 4)
        trip.total = trip.adults + trip.children
    else:  # group
        trip.total = rng.randint(7, 16)
        trip.adults = trip.total if _maybe(rng, .5) else None


def _sample_date(rng: random.Random) -> DateInfo:
    kind = rng.choices(
        ("month", "part_month", "range_month", "range_cross", "single", "numeric_ambiguous", "numeric_clear",
         "month_range", "month_choice"),
        weights=(26, 18, 16, 8, 8, 8, 6, 5, 5))[0]
    month = rng.randint(1, 12)
    d = DateInfo(kind=kind, month=month, month2=month % 12 + 1)
    d.year = rng.choice((None, None, None, 2027, 2027, 2028))
    d.part = rng.choice(("early", "mid", "late"))
    if kind == "range_month":
        d.day = rng.randint(1, 14)
        d.day2 = d.day + rng.randint(6, 13)
    elif kind == "range_cross":
        d.day = rng.randint(20, 28)
        d.day2 = rng.randint(2, 12)
    elif kind == "single":
        d.day = rng.randint(1, 28)
    elif kind == "numeric_ambiguous":
        d.day = rng.randint(1, 12)
        d.month = rng.choice([m for m in range(1, 13) if m != d.day])
    elif kind == "numeric_clear":
        d.day = rng.randint(13, 28)
    return d


def _nights(d: DateInfo) -> Optional[int]:
    if d.kind == "range_month":
        return d.day2 - d.day
    if d.kind == "range_cross":
        year = d.year or 2027
        start = dt.date(year, d.month, d.day)
        end = dt.date(year + (d.month2 < d.month), d.month2, d.day2)
        return (end - start).days
    return None


def _sample_duration(rng: random.Random) -> tuple[int, str]:
    unit = rng.choices(("nights", "days", "weeks"), weights=(5, 3, 3))[0]
    return (rng.randint(1, 3) if unit == "weeks" else rng.randint(5, 21), unit)


def _sample_budget(trip: Trip, rng: random.Random) -> Budget:
    weights = CURRENCY_WEIGHTS[trip.language]
    currency = rng.choices(list(weights), weights=list(weights.values()))[0]
    basis = rng.choices(("per_trip", "per_person", None), weights=(4, 3, 4))[0]
    base = rng.choice((3000, 4000, 5000, 6000, 7500, 9000)) if basis == "per_person" else \
        rng.choice((8000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 40000, 50000, 75000, 100000))
    if currency == "INR":
        base = base * 80 // 100000 * 100000 or 100000
    shape = rng.choices(("about", "exact", "up_to", "at_least", "range"), weights=(5, 2, 4, 1, 3))[0]
    if shape == "up_to":
        return Budget(None, base, currency, basis, shape)
    if shape == "at_least":
        return Budget(base, None, currency, basis, shape)
    if shape == "range":
        return Budget(base, int(base * rng.choice((1.25, 1.5, 2))), currency, basis, shape)
    return Budget(base, base, currency, basis, shape)


def sample_trip(language: str, style: str, rng: random.Random) -> Trip:
    persona = rng.choices(("couple", "family", "friends", "two_couples", "solo", "multigen", "group"),
                          weights=(38, 24, 8, 5, 11, 9, 5))[0]
    group = rng.choices(list(PLACE_GROUPS), weights=(3, 3, 3, 2, 2))[0]
    trip = Trip(language=language, style=style, persona=persona, group=group)
    chance = PRESENCE[style]

    if _maybe(rng, chance["dest"]):
        broad, towns = PLACE_GROUPS[group]
        how = rng.choice(("broad", "towns", "both"))
        if how == "broad":
            trip.places = _pick(rng, broad, 1, 2)
        elif how == "towns":
            trip.places = _pick(rng, towns, 1, 3)
        else:
            trip.places = _pick(rng, broad, 1, 1) + _pick(rng, towns, 1, 2)
        if group == "sri_lanka" and _maybe(rng, .2):
            trip.places = trip.places[:2] + ["Maldives"]

    if _maybe(rng, chance["party"]):
        _sample_people(trip, rng)
    if _maybe(rng, chance["date"]):
        trip.date = _sample_date(rng)
    if _maybe(rng, chance["dur"]):
        nights = _nights(trip.date) if trip.date else None
        trip.duration = (nights, "nights") if nights else _sample_duration(rng)
    if _maybe(rng, chance["budget"]):
        trip.budget = _sample_budget(trip, rng)
    if _maybe(rng, chance["accommodation"]):
        trip.accommodation = _pick(rng, GROUP_ACCOMMODATION[group], 1, 2)
        if persona in ("family", "multigen") and _maybe(rng, .5):
            trip.accommodation = trip.accommodation[:1] + [rng.choice(("family room", "connecting rooms"))]
    if _maybe(rng, chance["flights"]):
        if _maybe(rng, .25):
            trip.no_flights = True
        else:
            trip.flights = _pick(rng, FLIGHT_LABELS, 1, 2)
            if "economy class" in trip.flights and len(trip.flights) > 1:
                trip.flights = [f for f in trip.flights if f not in ("first class", "business class",
                                                                     "premium economy")] or trip.flights[:1]
            if _maybe(rng, .6):
                trip.origin = rng.choice(ORIGINS)
    if _maybe(rng, chance["interests"]):
        trip.interests = _pick(rng, GROUP_INTERESTS[group], 1, 3)
    if _maybe(rng, chance["dietary"]):
        trip.dietary = _pick(rng, DIETARY_LABELS, 1, 2)
    if _maybe(rng, chance["occasion"]):
        options = {"couple": COUPLE_OCCASIONS, "solo": SOLO_OCCASIONS}.get(persona, FAMILY_OCCASIONS)
        trip.occasion = rng.choice(options)
    if _maybe(rng, chance["special"]):
        options = list(GENERAL_SPECIAL)
        if trip.children:
            options += FAMILY_SPECIAL * 2
        if language == "fr":
            options.append("French-speaking guide")
        if language == "de":
            options.append("German-speaking guide")
        trip.special = _pick(rng, options, 1, 2)
        if trip.occasion in ("proposal", "birthday", "anniversary") and _maybe(rng, .4):
            trip.special.append("surprise arrangement")
    return trip
