"""Trained slot tagger: a CRF that finds field spans in an enquiry.

The CRF puts a BIO tag on every token of the project tokenizer, so each span keeps its exact
place in the text. It was trained on public data by training/train_crf.py and lives in
models/uc2_crf/ (git-ignored).

add_tagger_values() fills only what the rules left empty. Every value it adds is marked
source="crf", with the tagger's marginal probability as its confidence. Spans become values
through the existing code: parse_dates for dates and trip lengths, extract_travellers for
counts, extract_budget for money and currency, and the lexicons for labels. A date or length
that this code cannot read is flagged for a person, never guessed.

Known limits:
- French and German training data only has DATE, so only DATE can fill a field in those
  languages. The same holds when the language could not be detected.
- There is no training data for AGE, BUDGET, INTEREST, DIETARY, OCCASION or SPECIAL_REQUEST,
  so the tagger does not produce these labels. The BUDGET step below is ready for when it does.
- The marginal probability is the model's own number, not a calibrated probability.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

from pydantic import ValidationError

from . import lexicon as lx
from .dates import parse_dates
from .extract import (
    _ACCOMMODATION,
    _FLIGHT_CLASSES,
    _match_lexicon,
    _missing_flags,
    _negated,
    extract_budget,
    extract_destinations,
    extract_travellers,
)
from .schema import ClientProfile, Evidence, Flag, Provenance, TravelDates, Travellers
from .text import Token, detect_language, tokenize
from .training.convert_taskmaster import classify

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "uc2_crf"
MODEL_FILE = MODEL_DIR / "crf_tagger.crfsuite"
INFO_FILE = MODEL_DIR / "crf_tagger.json"
# Fixed in advance and not tuned: keep a span only if the tagger finds it more likely than not.
MIN_CONFIDENCE = 0.5

_LETTER = re.compile(r"[^\W\d_]")
_COUNT_FIELD = {"PARTY": "party_size", "ADULTS": "adults", "CHILDREN": "children"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def word_shape(word: str) -> str:
    """Letters become x or X and digits d; runs are squeezed: "Delhi" -> "Xx", "20,000" -> "d,d"."""
    shape = "".join("d" if c.isdigit() else ("X" if c.isupper() else "x") if c.isalpha() else c for c in word)
    return re.sub(r"(.)\1+", r"\1", shape)[:10]


def token_features(words: Sequence[str], language: Optional[str]) -> list[dict]:
    """Features of each token: word, prefix, suffix, shape, digit flag, neighbours and language."""
    low = [w.lower() for w in words]
    n = len(low)
    out = []
    for i, word in enumerate(low):
        f = {"bias": 1.0, "lang": language or "unknown", "w": word, "pre2": word[:2], "pre3": word[:3],
             "suf2": word[-2:], "suf3": word[-3:], "shape": word_shape(words[i]),
             "w-1": low[i - 1] if i >= 1 else "<s>", "w+1": low[i + 1] if i + 1 < n else "</s>",
             "w-2": low[i - 2] if i >= 2 else "<s>", "w+2": low[i + 2] if i + 2 < n else "</s>"}
        if any(c.isdigit() for c in word):
            f["digit"] = 1.0
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaggedSpan:
    label: str
    start: int
    end: int
    text: str
    confidence: float  # the lowest marginal probability among the span's tokens


class CrfTagger:
    """Wraps a CRFsuite model file. One tagger can only do one enquiry at a time, hence the lock."""

    def __init__(self, path: Path = MODEL_FILE, labels_by_language: Optional[dict] = None):
        if not Path(path).exists():
            raise FileNotFoundError(f"tagger model not found: {path}; run python -m src.uc2_extractor.training.train_crf")
        import pycrfsuite

        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(str(path))
        self._lock = threading.Lock()
        if labels_by_language is None:
            labels_by_language = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))["labels_by_language"]
        self.labels_by_language = {lang: set(labels) for lang, labels in labels_by_language.items()}

    def predict(self, words: Sequence[str], language: Optional[str]) -> tuple[list[str], list[float]]:
        """The best tag of each token, and its marginal probability."""
        features = token_features(words, language)
        with self._lock:
            self._tagger.set(features)
            tags = self._tagger.tag()
            return list(tags), [self._tagger.marginal(tag, i) for i, tag in enumerate(tags)]

    def labels_for(self, language: Optional[str]) -> set[str]:
        """Labels with enough training data in this language. With no language, those shared by all."""
        if language in self.labels_by_language:
            return self.labels_by_language[language]
        return set.intersection(*self.labels_by_language.values()) if self.labels_by_language else set()


@lru_cache(maxsize=1)
def default_tagger() -> CrfTagger:
    return CrfTagger()


def check_tagger() -> tuple[bool, str]:
    """Load the tagger once and report whether it can be used."""
    try:
        default_tagger().predict(["warm-up"], "en")
        return True, "Trained tagger available."
    except Exception as exc:  # missing file or package
        return False, f"Trained tagger not available ({type(exc).__name__})."


def spans_from_tags(text: str, tokens: Sequence[Token], tags: Sequence[str], marginals: Sequence[float]) -> list[TaggedSpan]:
    """Rebuild character spans from BIO tags, by the same rule as labels.bio_to_spans."""
    groups: list[dict] = []
    current = None
    for token, tag, score in zip(tokens, tags, marginals):
        prefix, _, label = tag.partition("-")
        if prefix == "B" or (prefix == "I" and (current is None or current["label"] != label)):
            current = {"label": label, "start": token.start, "end": token.end, "score": score}
            groups.append(current)
        elif prefix == "I":
            current["end"] = token.end
            current["score"] = min(current["score"], score)
        else:
            current = None
    return [TaggedSpan(g["label"], g["start"], g["end"], text[g["start"]:g["end"]], g["score"]) for g in groups]


def find_spans(text: str, language: Optional[str], tagger) -> list[TaggedSpan]:
    """Every span the tagger finds, with no filtering."""
    tokens = tokenize(text)
    if not tokens:
        return []
    tags, marginals = tagger.predict([t.text for t in tokens], language)
    return spans_from_tags(text, tokens, tags, marginals)


def usable_spans(text: str, language: Optional[str], tagger) -> list[TaggedSpan]:
    """Spans for labels this language has training data for, at MIN_CONFIDENCE or above, not negated."""
    allowed = tagger.labels_for(language)
    return [s for s in find_spans(text, language, tagger)
            if s.label in allowed and s.confidence >= MIN_CONFIDENCE and not _negated(text, s.start)]


# ---------------------------------------------------------------------------
# Spans to values
# ---------------------------------------------------------------------------


def _tidy(text: str) -> str:
    text = text.strip()
    return text.title() if text.islower() else text


def _place_values(text: str) -> list[str]:
    labels, _ = extract_destinations(text)
    if labels:
        return labels
    return [_tidy(text)] if _LETTER.search(text) else []


def _flight_value(text: str) -> Optional[str]:
    labels, _ = _match_lexicon(text, _FLIGHT_CLASSES, "flights")
    if labels:
        return labels[0]
    for name in ("flight_search.seating_class", "flight_search.stops"):
        parts = classify(name, text)
        if parts:
            return parts[0][3]
    return None


def _origin_value(text: str) -> Optional[str]:
    for city, forms in lx.FLIGHT_ORIGINS.items():
        if re.fullmatch("|".join(forms), text.strip(), re.IGNORECASE):
            return f"departure from {city}"
    return f"departure from {_tidy(text)}" if _LETTER.search(text) else None


def _count(label: str, text: str) -> Optional[int]:
    found, _, _ = extract_travellers(text)
    value = getattr(found, _COUNT_FIELD[label])
    if value is not None:
        return value
    for part_label, _, _, part_value in classify("hotel_search.num.guests", text) or []:
        if part_label == label:
            return part_value
    return None


def _is_valid(model, candidate) -> bool:
    try:
        model.model_validate(candidate.model_dump())
        return True
    except ValidationError:
        return False


class _Filler:
    """Adds tagger values to a profile and records where each one came from."""

    def __init__(self, profile: ClientProfile, text: str):
        self.p = profile
        self.text = text
        self.flags: list[Flag] = []

    def note(self, field: str, value: str, span: TaggedSpan) -> None:
        self.p.evidence.append(Evidence(field=field, text=span.text, start=span.start, end=span.end, source="crf"))
        confidence = round(min(max(span.confidence, 0.0), 1.0), 3)
        self.p.provenance.append(Provenance(field=field, value=value, source="crf", confidence=confidence))

    def add_to_list(self, field: str, value: Optional[str], span: TaggedSpan) -> None:
        values = getattr(self.p, field)
        if value and value not in values:
            values.append(value)
            self.note(field, value, span)

    def destinations(self, spans: Iterable[TaggedSpan]) -> None:
        if self.p.destination:
            return
        for s in spans:
            for value in _place_values(s.text):
                self.add_to_list("destination", value, s)

    def flights(self, cabin: Iterable[TaggedSpan], origin: Iterable[TaggedSpan]) -> None:
        if self.p.flights:
            return
        for s in cabin:
            self.add_to_list("flights", _flight_value(s.text), s)
        for s in origin:
            self.add_to_list("flights", _origin_value(s.text), s)

    def accommodation(self, spans: Iterable[TaggedSpan]) -> None:
        if self.p.accommodation:
            return
        for s in spans:  # the field holds fixed labels, so a span with no label adds nothing
            for label in _match_lexicon(s.text, _ACCOMMODATION, "accommodation")[0]:
                self.add_to_list("accommodation", label, s)

    def unreadable(self, field: str, span: TaggedSpan, what: str) -> None:
        self.flags.append(Flag(field=field, kind="ambiguous", message=f"'{span.text}' looks like {what} but could not be read"))

    def _set_dates(self, **values) -> bool:
        candidate = self.p.dates.model_copy(update=values)
        if not _is_valid(TravelDates, candidate):
            return False
        self.p.dates = candidate
        return True

    def dates(self, spans: Iterable[TaggedSpan]) -> None:
        d = self.p.dates
        if d.start or d.end or d.ambiguous:
            return
        for s in spans:
            result = parse_dates(s.text)
            got = result.dates
            d = self.p.dates
            if got.ambiguous:
                if self._set_dates(ambiguous=d.ambiguous + got.ambiguous):
                    self.note("dates", "ambiguous", s)
                    self.flags += [f for f in result.flags if f.kind == "ambiguous"]
            elif got.start is None and got.end is None:
                self.unreadable("dates", s, "a date")
            elif d.start is None and d.end is None:
                if self._set_dates(start=got.start, end=got.end):
                    self.note("dates", "start and end" if got.start and got.end else "start" if got.start else "end", s)
            elif d.end is None and got.start and got.start != d.start:
                if self._set_dates(end=got.start):  # a second date closes the range
                    self.note("dates", "end", s)

    def duration(self, spans: Iterable[TaggedSpan]) -> None:
        for s in spans:
            if self.p.dates.duration is not None:
                return
            found = parse_dates(s.text).dates.duration
            if found is None:
                self.unreadable("dates", s, "a trip length")
            elif self._set_dates(duration=found):
                self.note("dates", "duration", s)

    def travellers(self, spans: Iterable[TaggedSpan]) -> None:
        last = None
        for s in spans:
            field = _COUNT_FIELD[s.label]
            value = _count(s.label, s.text)
            if value is None or getattr(self.p.travellers, field) is not None:
                continue
            candidate = self.p.travellers.model_copy(update={field: value})
            if _is_valid(Travellers, candidate):
                self.p.travellers = candidate
                self.note("travellers", field, s)
                last = s
        t = self.p.travellers
        if last and t.party_size is None and t.adults is not None:  # same rule as extract_travellers
            self.p.travellers = t.model_copy(update={"party_size": t.adults + (t.children or 0)})
            self.note("travellers", "party_size", last)

    def budget(self, spans: Iterable[TaggedSpan]) -> None:
        if self.p.budget.amount_min is not None or self.p.budget.amount_max is not None:
            return
        for s in spans:
            found, _, flags = extract_budget(s.text)
            if found.amount_min is not None or found.amount_max is not None:
                self.p.budget = found
                self.flags += flags
                self.note("budget", "", s)
                return


def add_tagger_values(profile: ClientProfile, text: str, tagger=None) -> ClientProfile:
    """Fill what the rules left empty, marking every value source="crf"."""
    tagger = tagger or default_tagger()
    missing_before = _missing_flags(profile)
    other_flags = [f for f in profile.flags if f not in missing_before]

    by_label: dict[str, list[TaggedSpan]] = {}
    for s in usable_spans(text, profile.language, tagger):
        by_label.setdefault(s.label, []).append(s)

    fill = _Filler(profile, text)
    fill.destinations(by_label.get("DESTINATION", []))
    fill.flights(by_label.get("FLIGHT", []), by_label.get("ORIGIN", []))
    fill.accommodation(by_label.get("ACCOMMODATION", []))
    fill.dates(by_label.get("DATE", []))
    fill.duration(by_label.get("DURATION", []))
    fill.travellers([s for s in sorted(sum((by_label.get(k, []) for k in _COUNT_FIELD), []), key=lambda x: x.start)])
    fill.budget(by_label.get("BUDGET", []))

    profile.evidence.sort(key=lambda e: (e.start, e.field))
    profile.flags = _missing_flags(profile) + other_flags + fill.flags
    return profile


def tagger_profile(text: str, tagger=None) -> ClientProfile:
    """The tagger alone, with no rule values. Used to score it, not offered in the app."""
    profile = ClientProfile(language=detect_language(text))
    profile.flags = _missing_flags(profile)
    return add_tagger_values(profile, text, tagger)
