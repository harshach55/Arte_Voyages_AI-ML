"""Field-level evaluation of extracted profiles against gold labels.

Values are compared as text in the same format as the gold spreadsheet:
lists are joined with " | ", dates use PartialDate text (2027-03-04, --03-04,
--03), durations look like "14 nights".

Exact match: the normalised values are equal.
Lenient match:
- text and list items: one contains the other ("Pangong" vs "Pangong Lake")
- dates: same month, and day/year equal where both are given (--03 vs --03-03)
- durations: within one day of each other (14 nights vs 14 days)
- numbers: no leniency (a wrong count or amount is wrong)

The evaluation is value-based, not span-based: the gold set records the
expected values, not character positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import ClientProfile, PartialDate

LIST_FIELDS = (
    "destination", "dates_ambiguous", "ages", "accommodation", "flights", "interests",
    "dietary", "occasion", "special_requests",
)
SCALAR_FIELDS = (
    "language", "start_date", "end_date", "duration", "party_size", "adults", "children",
    "budget_min", "budget_max", "currency", "budget_basis", "budget_band",
)
ALL_FIELDS = (
    "language", "destination", "start_date", "end_date", "duration", "dates_ambiguous",
    "party_size", "adults", "children", "ages", "budget_min", "budget_max", "currency",
    "budget_basis", "budget_band", "accommodation", "flights", "interests", "dietary",
    "occasion", "special_requests",
)
DATE_FIELDS = ("start_date", "end_date")
NUMBER_FIELDS = ("party_size", "adults", "children", "ages", "budget_min", "budget_max")
SEP = " | "


# ---------------------------------------------------------------------------
# Profile -> flat text values
# ---------------------------------------------------------------------------


def _num(v) -> str:
    if v is None:
        return ""
    return str(int(v)) if float(v).is_integer() else str(v)


def profile_to_values(p: ClientProfile) -> dict[str, str]:
    d, t, b = p.dates, p.travellers, p.budget
    return {
        "language": p.language or "",
        "destination": SEP.join(p.destination),
        "start_date": d.start.to_str() if d.start else "",
        "end_date": d.end.to_str() if d.end else "",
        "duration": d.duration.to_str() if d.duration else "",
        "dates_ambiguous": SEP.join(a.text for a in d.ambiguous),
        "party_size": _num(t.party_size),
        "adults": _num(t.adults),
        "children": _num(t.children),
        "ages": SEP.join(str(a) for a in t.ages),
        "budget_min": _num(b.amount_min),
        "budget_max": _num(b.amount_max),
        "currency": b.currency or "",
        "budget_basis": b.basis or "",
        "budget_band": b.band or "",
        "accommodation": SEP.join(p.accommodation),
        "flights": SEP.join(p.flights),
        "interests": SEP.join(p.interests),
        "dietary": SEP.join(p.dietary),
        "occasion": SEP.join(p.occasion),
        "special_requests": SEP.join(p.special_requests),
    }


# ---------------------------------------------------------------------------
# Normalisation and matching
# ---------------------------------------------------------------------------


def normalise(value: str, field_name: str) -> str:
    v = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if field_name in NUMBER_FIELDS and v:
        try:
            v = _num(float(v))
        except ValueError:
            pass
    return v


def split_list(value: str, field_name: str) -> list[str]:
    return [normalise(x, field_name) for x in str(value or "").split("|") if x.strip()]


def _date_lenient(a: str, b: str) -> bool:
    try:
        x, y = PartialDate.from_str(a), PartialDate.from_str(b)
    except (ValueError, IndexError):
        return False
    if x.month != y.month:
        return False
    if x.day is not None and y.day is not None and x.day != y.day:
        return False
    if x.year is not None and y.year is not None and x.year != y.year:
        return False
    return True


def _duration_days(v: str) -> Optional[int]:
    m = re.fullmatch(r"(\d+) (nights|days|weeks)", v)
    if not m:
        return None
    n = int(m.group(1))
    return n * 7 if m.group(2) == "weeks" else n


def values_match(gold: str, pred: str, field_name: str, lenient: bool) -> bool:
    """Compare two already-normalised single values."""
    if gold == pred:
        return True
    if not lenient or not gold or not pred:
        return False
    if field_name in NUMBER_FIELDS:
        return False
    if field_name in DATE_FIELDS:
        return _date_lenient(gold, pred)
    if field_name == "duration":
        g, p = _duration_days(gold), _duration_days(pred)
        return g is not None and p is not None and abs(g - p) <= 1
    return gold in pred or pred in gold


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    extra_on_empty: int = 0  # a value predicted where the gold field is empty

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.extra_on_empty += other.extra_on_empty

    @property
    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class FieldResult:
    counts: Counts
    missed: list[str] = field(default_factory=list)  # gold values not found
    extra: list[str] = field(default_factory=list)  # predicted values not in gold


def compare_field(gold: str, pred: str, field_name: str, lenient: bool = False) -> FieldResult:
    """Compare one field of one enquiry."""
    c = Counts()
    if field_name in LIST_FIELDS:
        g_items = split_list(gold, field_name)
        p_items = split_list(pred, field_name)
        unmatched = list(g_items)
        extra = []
        for p in p_items:
            hit = next((g for g in unmatched if values_match(g, p, field_name, lenient)), None)
            if hit is None:
                extra.append(p)
            else:
                unmatched.remove(hit)
                c.tp += 1
        c.fp, c.fn = len(extra), len(unmatched)
        if not g_items and p_items:
            c.extra_on_empty = 1
        return FieldResult(c, missed=unmatched, extra=extra)

    g, p = normalise(gold, field_name), normalise(pred, field_name)
    if not g and not p:
        return FieldResult(c)
    if g and p and values_match(g, p, field_name, lenient):
        c.tp = 1
        return FieldResult(c)
    if p:
        c.fp = 1
        if not g:
            c.extra_on_empty = 1
    if g:
        c.fn = 1
    return FieldResult(c, missed=[g] if g else [], extra=[p] if p else [])


def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.2f}"
