"""Client profile schema for the travel enquiry extractor.

This is the single definition of the output of the extractor. Other code
should import ClientProfile from here instead of redefining the fields.

Rules that the schema enforces:
- every field may be empty; an empty field means "not found in the text"
- numbers and dates must be valid (no 30 February, no negative ages)
- a budget band is only allowed for USD budgets that are not per person
- every piece of evidence must point at a real slice of the enquiry
"""

from __future__ import annotations

import calendar
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LANGUAGES = ("en", "fr", "de")

FIELD_NAMES = (
    "language",
    "destination",
    "dates",
    "travellers",
    "budget",
    "accommodation",
    "flights",
    "interests",
    "dietary",
    "occasion",
    "special_requests",
)

BANDS_FILE = Path(__file__).with_name("budget_bands.toml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Budget bands (loaded from budget_bands.toml)
# ---------------------------------------------------------------------------


class BudgetBand(_Strict):
    label: str
    min: Optional[float] = None  # inclusive
    max: Optional[float] = None  # exclusive


class BandConfig(_Strict):
    currency: str
    basis: Literal["per_trip", "per_person"]
    placeholder: bool = False
    band: list[BudgetBand]

    @model_validator(mode="after")
    def _bands_are_contiguous(self) -> "BandConfig":
        bands = self.band
        if not bands:
            raise ValueError("at least one budget band is needed")
        if bands[0].min is not None or bands[-1].max is not None:
            raise ValueError("first band must have no min and last band no max")
        for low, high in zip(bands, bands[1:]):
            if low.max is None or low.max != high.min:
                raise ValueError(f"bands {low.label} and {high.label} do not touch")
        labels = [b.label for b in bands]
        if len(set(labels)) != len(labels):
            raise ValueError("band labels must be unique")
        return self


@lru_cache(maxsize=None)
def load_band_config(path: Path = BANDS_FILE) -> BandConfig:
    with open(path, "rb") as f:
        return BandConfig.model_validate(tomllib.load(f))


def band_for_amount(amount: float, config: Optional[BandConfig] = None) -> str:
    """Return the band label for an amount already in the band currency."""
    config = config or load_band_config()
    for band in config.band:
        above_min = band.min is None or amount >= band.min
        below_max = band.max is None or amount < band.max
        if above_min and below_max:
            return band.label
    raise ValueError(f"no band for amount {amount}")  # unreachable for valid config


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Evidence(_Strict):
    """A slice of the enquiry text that supports an extracted value."""

    field: str
    text: str
    start: int = Field(ge=0)
    end: int

    @field_validator("field")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in FIELD_NAMES:
            raise ValueError(f"unknown field {v!r}")
        return v

    @model_validator(mode="after")
    def _span_matches_text(self) -> "Evidence":
        if self.end <= self.start:
            raise ValueError("evidence end must be after start")
        if self.end - self.start != len(self.text):
            raise ValueError("evidence text length does not match its span")
        return self


class Flag(_Strict):
    """Something a person should look at: a missing, ambiguous or assumed value."""

    field: str
    kind: Literal["missing", "ambiguous", "assumption"]
    message: str

    @field_validator("field")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in FIELD_NAMES:
            raise ValueError(f"unknown field {v!r}")
        return v


class PartialDate(_Strict):
    """A calendar date where the year and day may be unknown.

    Written as text in ISO-like form: 2027-03-04, --03-04 (no year),
    2027-03 or --03 (month only).
    """

    year: Optional[int] = Field(default=None, ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    day: Optional[int] = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _day_exists(self) -> "PartialDate":
        if self.day is not None:
            # with no year, allow 29 February (it exists in leap years)
            year = self.year if self.year is not None else 2028
            last = calendar.monthrange(year, self.month)[1]
            if self.day > last:
                raise ValueError(f"day {self.day} does not exist in month {self.month}")
        return self

    def to_str(self) -> str:
        prefix = f"{self.year:04d}" if self.year is not None else "-"
        text = f"{prefix}-{self.month:02d}"
        if self.day is not None:
            text += f"-{self.day:02d}"
        return text

    @classmethod
    def from_str(cls, text: str) -> "PartialDate":
        text = text.strip()
        if text.startswith("--"):
            parts = text[2:].split("-")
            year = None
        else:
            year_text, *parts = text.split("-")
            year = int(year_text)
        month = int(parts[0])
        day = int(parts[1]) if len(parts) > 1 else None
        return cls(year=year, month=month, day=day)


class Duration(_Strict):
    value: int = Field(ge=1, le=365)
    unit: Literal["nights", "days", "weeks"]

    def to_str(self) -> str:
        return f"{self.value} {self.unit}"


class AmbiguousDate(_Strict):
    """A date written so that it has more than one reading, e.g. 3/4."""

    text: str
    readings: list[PartialDate] = Field(min_length=2)


# ---------------------------------------------------------------------------
# Profile sections
# ---------------------------------------------------------------------------


class TravelDates(_Strict):
    start: Optional[PartialDate] = None
    end: Optional[PartialDate] = None
    duration: Optional[Duration] = None
    ambiguous: list[AmbiguousDate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "TravelDates":
        s, e = self.start, self.end
        # only checkable when both dates are complete; without a year a
        # range such as 20 Dec - 5 Jan is legitimate
        if s and e and None not in (s.year, s.day, e.year, e.day):
            if (e.year, e.month, e.day) < (s.year, s.month, s.day):
                raise ValueError("end date is before start date")
        return self


class Travellers(_Strict):
    party_size: Optional[int] = Field(default=None, ge=1, le=100)
    adults: Optional[int] = Field(default=None, ge=0, le=100)
    children: Optional[int] = Field(default=None, ge=0, le=100)
    ages: list[int] = Field(default_factory=list)

    @field_validator("ages")
    @classmethod
    def _ages_in_range(cls, v: list[int]) -> list[int]:
        for age in v:
            if not 0 <= age <= 120:
                raise ValueError(f"age {age} is out of range")
        return v

    @model_validator(mode="after")
    def _counts_agree(self) -> "Travellers":
        if None not in (self.party_size, self.adults, self.children):
            if self.adults + self.children != self.party_size:
                raise ValueError("adults + children does not equal party size")
        if self.party_size is not None and len(self.ages) > self.party_size:
            raise ValueError("more ages than travellers")
        return self


class Budget(_Strict):
    amount_min: Optional[float] = Field(default=None, ge=0)
    amount_max: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    basis: Optional[Literal["per_trip", "per_person"]] = None
    band: Optional[str] = None

    @model_validator(mode="after")
    def _consistent(self) -> "Budget":
        if None not in (self.amount_min, self.amount_max):
            if self.amount_min > self.amount_max:
                raise ValueError("budget minimum is above maximum")
        if self.band is not None:
            config = load_band_config()
            if self.band not in {b.label for b in config.band}:
                raise ValueError(f"unknown budget band {self.band!r}")
            if self.currency != config.currency:
                raise ValueError("a band is only allowed in the band currency (no conversion)")
            if self.basis == "per_person":
                raise ValueError("bands are per trip; a per-person budget gets no band")
        return self


class ClientProfile(_Strict):
    """Validated client profile extracted from one enquiry."""

    language: Optional[Literal["en", "fr", "de"]] = None
    destination: list[str] = Field(default_factory=list)
    dates: TravelDates = Field(default_factory=TravelDates)
    travellers: Travellers = Field(default_factory=Travellers)
    budget: Budget = Field(default_factory=Budget)
    accommodation: list[str] = Field(default_factory=list)
    flights: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    dietary: list[str] = Field(default_factory=list)
    occasion: list[str] = Field(default_factory=list)
    # special requests and must-haves share one field
    special_requests: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)


def check_evidence(profile: ClientProfile, text: str) -> list[str]:
    """Return a list of problems where evidence does not match the enquiry text."""
    problems = []
    for ev in profile.evidence:
        if ev.end > len(text) or text[ev.start:ev.end] != ev.text:
            problems.append(f"{ev.field}: {ev.text!r} not found at {ev.start}-{ev.end}")
    return problems
