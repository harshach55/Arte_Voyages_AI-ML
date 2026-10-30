"""Rule-based extraction of a client profile from one travel enquiry.

Use extract_profile(text) to get a validated ClientProfile. Every value comes
with the slice of text it was taken from. Anything the rules cannot find
stays empty, and the most important gaps are flagged.
"""

from __future__ import annotations

import re
from typing import Optional

from . import lexicon as lx
from .dates import parse_dates
from .schema import (
    Budget,
    ClientProfile,
    Evidence,
    Flag,
    Travellers,
    band_for_amount,
    load_band_config,
)
from .text import detect_language

_I = re.IGNORECASE


def _compile(lexicon: dict[str, list[str]]) -> dict[str, re.Pattern]:
    return {
        label: re.compile(rf"(?<!\w)(?:{'|'.join(frags)})(?!\w)", _I)
        for label, frags in lexicon.items()
    }


_DESTINATIONS = _compile(lx.DESTINATIONS)
_ACCOMMODATION = _compile(lx.ACCOMMODATION)
_FLIGHT_CLASSES = _compile(lx.FLIGHT_CLASSES)
_INTERESTS = _compile(lx.INTERESTS)
_DIETARY = _compile(lx.DIETARY)
_OCCASIONS = _compile(lx.OCCASIONS)
_SPECIAL = _compile(lx.SPECIAL_REQUESTS)

# ---------------------------------------------------------------------------
# Negation and lexicon matching
# ---------------------------------------------------------------------------

_CLAUSE_BREAK = re.compile(r"[.;:!?\n,()]|\b(?:but|mais|aber|however|sondern)\b", _I)


def _negated(text: str, start: int) -> bool:
    """True if a negation word appears earlier in the same clause (up to 40 chars)."""
    window = text[max(0, start - 40):start]
    tail = _CLAUSE_BREAK.split(window)[-1]
    return bool(lx.NEGATION_RE.search(tail))


def _ev(field: str, text: str, start: int, end: int) -> Evidence:
    return Evidence(field=field, text=text[start:end], start=start, end=end)


def _match_lexicon(text: str, compiled: dict[str, re.Pattern], field: str):
    """Find lexicon labels in the text. Longer matches win over shorter
    overlapping ones (e.g. 'water villa' beats 'villa'). Negated matches are
    dropped."""
    hits = [
        (m.start(), m.end(), label)
        for label, rx in compiled.items()
        for m in rx.finditer(text)
    ]
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    taken: list[tuple[int, int]] = []
    kept = []
    for s, e, label in hits:
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        if not _negated(text, s):
            kept.append((s, e, label))
    kept.sort()
    labels = list(dict.fromkeys(label for _, _, label in kept))
    return labels, [_ev(field, text, s, e) for s, e, _ in kept]


# ---------------------------------------------------------------------------
# Flights
# ---------------------------------------------------------------------------

_FLIGHT_WORD_RE = re.compile(rf"(?<!\w)(?:{lx.FLIGHT_WORDS})(?!\w)", _I)
_ORIGINS = {
    city: re.compile(
        rf"(?<!\w)(?:{lx.ORIGIN_PREPOSITIONS})\s*(?:{'|'.join(frags)})(?!\w)", _I
    )
    for city, frags in lx.FLIGHT_ORIGINS.items()
}
_SENTENCE_RE = re.compile(r"[^.!?\n]+")


def extract_flights(text: str):
    labels, evidence = _match_lexicon(text, _FLIGHT_CLASSES, "flights")
    # a departure city only counts in a sentence that talks about flying
    for sent in _SENTENCE_RE.finditer(text):
        if not _FLIGHT_WORD_RE.search(sent.group()):
            continue
        for city, rx in _ORIGINS.items():
            for m in rx.finditer(sent.group()):
                s, e = sent.start() + m.start(), sent.start() + m.end()
                if not _negated(text, s):
                    label = f"departure from {city}"
                    if label not in labels:
                        labels.append(label)
                    evidence.append(_ev("flights", text, s, e))
    return labels, sorted(evidence, key=lambda x: x.start)


# ---------------------------------------------------------------------------
# Travellers
# ---------------------------------------------------------------------------

_NUMW = lx.alternation(lx.NUMBER_WORDS)
_N = rf"(?P<n>\d{{1,2}}|{_NUMW})"
_UNIT_AFTER = r"(?!\s*(?:weeks?|wochen|semaines?|nights?|n[äa]chte|nuits?|days?|tage|jours?|years?|ans|jahre))"

_ADULTS_RE = re.compile(rf"(?<!\w){_N}\s+(?:adults?|adultes?|erwachsene[nr]?|grown-ups)(?!\w)", _I)
_GENERIC_CHILD = (
    r"children|child|kids?|enfants?|kinder[n]?|teenagers?|teens|adolescents?|jugendliche[n]?|"
    r"grandchildren|petits-enfants|enkel(?:kinder[n]?)?"
)
_SPECIFIC_CHILD = r"sons?|daughters?|fils|filles?|s[öo]hne[n]?|sohn|t[öo]chter[n]?|tochter|grandsons?|granddaughters?"
_CHILDREN_RE = re.compile(
    rf"(?<!\w){_N}\s+(?:(?:young|little|small|petits?|kleine[n]?)\s+)?"
    rf"(?P<noun>{_GENERIC_CHILD}|{_SPECIFIC_CHILD})(?!\w)",
    _I,
)
_GENERIC_CHILD_RE = re.compile(rf"^(?:{_GENERIC_CHILD})$", _I)
# one child named without a number: "our son", "notre fille", "mit unserem Baby"
_ONE_CHILD_RE = re.compile(
    r"(?<!\w)(?:a|an|our|un|une|notre|ein(?:em|en)?|unser(?:em|en)?)\s+"
    r"(?:baby|b[ée]b[ée]|infant|nourrisson|kleinkind|son|daughter|child|fils|fille|enfant|sohn|tochter|kind)(?!\w)",
    _I,
)
_TOTAL_RES = [
    re.compile(rf"(?:party|group|groupe|gruppe|family|famille|familie) (?:of|de|von) {_N}(?!\w)", _I),
    re.compile(
        rf"(?<!\w){_N}\s+(?:people|persons|guests|travell?ers|pax|personnes|voyageurs|personen|"
        rf"reisende[n]?|g[äa]ste[n]?|teilnehmer)(?!\w)",
        _I,
    ),
    re.compile(
        rf"(?:we are|we're|we will be|there are|there will be|nous sommes|nous serons|wir sind|wir werden)"
        rf"\s+{_N}(?!\w){_UNIT_AFTER}",
        _I,
    ),
    re.compile(rf"(?<!\w){_N}k[öo]pfige?\w*", _I),
]
_COUPLE_RE = re.compile(
    r"(?:my|our)\s+(?:wife|husband|partner|fianc[ée]e?|girlfriend|boyfriend|spouse)\s+and\s+I(?!\w)"
    r"|(?:ma femme|mon mari|mon [ée]pouse|mon [ée]poux|mon compagnon|ma compagne|mon partenaire|"
    r"ma partenaire|mon conjoint|ma conjointe|mon fianc[ée]|ma fianc[ée]e)\s+et\s+moi"
    r"|(?:meine (?:ehe)?frau|mein (?:ehe)?mann|mein partner|meine partnerin|meine freundin|mein freund|"
    r"mein verlobter|meine verlobte)\s+und\s+ich"
    r"|as a couple|the two of us|en couple|nous deux|tous les deux|zu zweit|wir beide",
    _I,
)
_SOLO_RE = re.compile(
    r"travell?ing (?:alone|solo)|solo travell?er|on my own|by myself|voyage(?:r|rai)? seule?"
    r"|seule? voyageu(?:r|se)|allein(?:e)? reisen|alleinreisende?r?|reise allein",
    _I,
)
_NUMLIST = r"\d{1,2}(?:\s*(?:,|and|&|et|und)\s*\d{1,2})*"
_AGE_RES = [
    re.compile(rf"(?:[âa]g[ée]e?s?)\s+de\s+(?P<list>{_NUMLIST})\s+ans(?!\w)", _I),
    re.compile(rf"\b(?:aged?|ages)\s+(?P<list>{_NUMLIST})", _I),
    re.compile(rf"(?P<list>{_NUMLIST})\s*(?:-|\s)years?(?:-|\s)olds?(?!\w)", _I),
    re.compile(rf"im alter von\s+(?P<list>{_NUMLIST})", _I),
    re.compile(rf"(?P<list>{_NUMLIST})\s+jahre(?:n)?\s+alt(?!\w)", _I),
    re.compile(r"(?<!\d)(?P<list>\d{1,2})-?j[äa]hrig\w*", _I),
    re.compile(rf"(?<!\d)(?P<list>{_NUMLIST})\s+ans(?!\w)(?!\s+de\s+mariage)", _I),
]
# "fête ses 60 ans", "depuis 20 ans": a number of years, not a traveller's age
_NOT_AGE_BEFORE = re.compile(r"\b(?:ses|mes|nos|son|ces|depuis|pendant|il y a|tous les)\s+$", _I)


def _num(token: str) -> int:
    return int(token) if token.isdigit() else lx.NUMBER_WORDS[token.lower()]


def extract_travellers(text: str):
    evidence: list[Evidence] = []
    flags: list[Flag] = []

    adults = None
    m = _ADULTS_RE.search(text)
    if m:
        adults = _num(m.group("n"))
        evidence.append(_ev("travellers", text, m.start(), m.end()))
    else:
        for rx, value in ((_COUPLE_RE, 2), (_SOLO_RE, 1)):
            m = rx.search(text)
            if m:
                adults = value
                evidence.append(_ev("travellers", text, m.start(), m.end()))
                break

    children = None
    child_hits = list(_CHILDREN_RE.finditer(text))
    generic = [h for h in child_hits if _GENERIC_CHILD_RE.match(h.group("noun"))]
    if generic:
        children = _num(generic[0].group("n"))
        evidence.append(_ev("travellers", text, generic[0].start(), generic[0].end()))
    elif child_hits:  # e.g. "two sons and one daughter"
        children = sum(_num(h.group("n")) for h in child_hits)
        evidence += [_ev("travellers", text, h.start(), h.end()) for h in child_hits]
    if not generic:
        for b in _ONE_CHILD_RE.finditer(text):
            if any(b.start() < h.end() and h.start() < b.end() for h in child_hits):
                continue  # already counted, e.g. "une fille"
            children = (children or 0) + 1
            evidence.append(_ev("travellers", text, b.start(), b.end()))

    total = None
    for rx in _TOTAL_RES:
        m = rx.search(text)
        if m:
            total = _num(m.group("n"))
            evidence.append(_ev("travellers", text, m.start(), m.end()))
            break

    ages: list[int] = []
    taken: list[tuple[int, int]] = []
    for rx in _AGE_RES:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            if _NOT_AGE_BEFORE.search(text[max(0, m.start() - 12):m.start()]):
                continue
            taken.append((m.start(), m.end()))
            evidence.append(_ev("travellers", text, m.start(), m.end()))
    for s, e in sorted(taken):
        ages += [int(x) for x in re.findall(r"\d{1,2}", text[s:e])]

    party = total
    if party is None and adults is not None:
        party = adults + (children or 0)
    if None not in (total, adults, children) and adults + children != total:
        flags.append(
            Flag(field="travellers", kind="ambiguous",
                 message=f"{adults} adults + {children} children does not match {total} travellers")
        )
        adults = children = None
    if party is not None and len(ages) > party:
        flags.append(Flag(field="travellers", kind="ambiguous",
                          message=f"{len(ages)} ages found for {party} travellers"))
        ages = []

    t = Travellers(party_size=party, adults=adults, children=children, ages=ages)
    return t, sorted(evidence, key=lambda x: x.start), flags


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

_CURRENCY_CODES = {
    code: re.compile(rf"(?:{'|'.join(frags)})", _I) for code, frags in lx.CURRENCIES.items()
}
_CUR = (
    r"(?<![^\W\d_])(?:"
    + "|".join(f for frags in lx.CURRENCIES.values() for f in sorted(frags, key=len, reverse=True))
    + r")(?![^\W\d_])"
)
_AMOUNT = r"\d{1,3}(?:[   ,.'’]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
_AMT = rf"(?:{_AMOUNT})(?:\s*(?:k|thousand|mille|tausend|tsd\.?)(?![^\W\d_]))?"
_TO = r"\s*(?:-|–|to|à|bis|and|et|und)\s*"

_BUDGET_RES = [
    ("range", re.compile(
        rf"(?:(?P<c1>{_CUR})\s*)?(?P<a1>{_AMT})\s*(?P<c2>{_CUR})?{_TO}(?:(?P<c3>{_CUR})\s*)?"
        rf"(?P<a2>{_AMT})(?:\s*(?P<c4>{_CUR}))?", _I)),
    ("single", re.compile(rf"(?P<c1>{_CUR})\s*(?P<a1>{_AMT})|(?P<a2>{_AMT})\s*(?P<c2>{_CUR})", _I)),
    ("keyword", re.compile(rf"\bbudg[eé]t\w*\D{{0,25}}?(?P<a1>{_AMT})", _I)),
]
_BUDGET_WORD = re.compile(r"\bbudg[eé]t", _I)
_UP_TO = re.compile(
    r"(?:up to|max(?:imum)?|no more than|not more than|under|below|less than|jusqu'[àa]|pas plus de|"
    r"moins de|au maximum|bis zu|h[öo]chstens|maximal|unter|weniger als)\s*$",
    _I,
)
_AT_LEAST = re.compile(
    r"(?:at least|minimum|from|starting at|au moins|[àa] partir de|mindestens|ab)\s*$", _I
)
_RATE_AFTER = re.compile(r"^\s*(?:per|a|an|par|pro|la|die)\s+(?:night|nuit|nacht|day|jour|tag)\b", _I)
_PER_PERSON = re.compile(
    r"\b(?:per person|per head|pp|p\.\s?p\.|each|par personne|par pers\.?|pro person|pro kopf)(?!\w)", _I
)
_PER_TRIP = re.compile(
    r"\b(?:in total|total|for (?:the )?(?:whole|entire) (?:trip|group|family)|for all of us|for everyone|"
    r"au total|en tout|pour (?:tout le monde|nous (?:deux|tous))|insgesamt|gesamt|f[üu]r alle)(?!\w)",
    _I,
)


def _parse_amount(raw: str) -> float:
    s = raw.strip().lower()
    mult = 1
    m = re.search(r"\s*(?:k|thousand|mille|tausend|tsd\.?)$", s)
    if m:
        mult, s = 1000, s[:m.start()]
    if re.fullmatch(r"\d{1,3}(?:[   ,.'’]\d{3})+", s):
        value = float(re.sub(r"\D", "", s))
    else:
        dm = re.fullmatch(r"(.*?)[.,](\d{1,2})", s)
        if dm:
            value = float(re.sub(r"\D", "", dm.group(1)) + "." + dm.group(2))
        else:
            value = float(re.sub(r"\D", "", s))
    return value * mult


def _currency_code(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    for code, rx in _CURRENCY_CODES.items():
        if rx.fullmatch(raw.strip()):
            return code
    return None


def extract_budget(text: str):
    flags: list[Flag] = []
    mentions = []
    taken: list[tuple[int, int]] = []
    for kind, rx in _BUDGET_RES:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            if _RATE_AFTER.search(text[m.end():m.end() + 15]):
                continue  # a nightly rate, not the trip budget
            g = m.groupdict()
            cur = next((g[c] for c in ("c1", "c2", "c3", "c4") if g.get(c)), None)
            if kind == "range" and not cur and not _BUDGET_WORD.search(text[max(0, m.start() - 40):m.start()]):
                continue  # plain number range, e.g. "8 and 11"
            if kind == "range":
                lo, hi = _parse_amount(g["a1"]), _parse_amount(g["a2"])
                if lo >= hi:
                    continue
                amount_min, amount_max = lo, hi
            else:
                value = _parse_amount(g["a1"] or g["a2"])
                amount_min = amount_max = value
                before = text[max(0, m.start() - 25):m.start()]
                if _UP_TO.search(before):
                    amount_min = None
                elif _AT_LEAST.search(before):
                    amount_max = None
            taken.append((m.start(), m.end()))
            mentions.append(dict(start=m.start(), end=m.end(), cur=cur,
                                 amount_min=amount_min, amount_max=amount_max))

    if not mentions:
        return Budget(), [], flags

    def score(x):
        near_word = bool(_BUDGET_WORD.search(text[max(0, x["start"] - 40):x["end"] + 10]))
        return (x["cur"] is not None, near_word, -x["start"])

    best = max(mentions, key=score)
    code = _currency_code(best["cur"])
    if code == "USD?":
        code = "USD"
        flags.append(Flag(field="budget", kind="assumption",
                          message=f"'{best['cur']}' read as US dollars; please confirm the currency"))
    if code is None:
        flags.append(Flag(field="budget", kind="missing", message="budget currency not stated"))

    around = text[max(0, best["start"] - 30):best["end"] + 40]
    basis = None
    if _PER_PERSON.search(around):
        basis = "per_person"
    elif _PER_TRIP.search(around):
        basis = "per_trip"

    band = None
    config = load_band_config()
    if code == config.currency and basis != "per_person":
        values = [v for v in (best["amount_min"], best["amount_max"]) if v is not None]
        bands = {band_for_amount(v, config) for v in values}
        if len(bands) == 1:
            band = bands.pop()
            if basis is None:
                flags.append(Flag(field="budget", kind="assumption",
                                  message="budget assumed to be for the whole trip"))
        else:
            flags.append(Flag(field="budget", kind="ambiguous",
                              message="budget range spans more than one band"))
    elif basis == "per_person":
        flags.append(Flag(field="budget", kind="missing",
                          message="budget is per person; per-trip band not assigned"))
    elif code is not None:
        flags.append(Flag(field="budget", kind="missing",
                          message=f"budget is in {code}; band not assigned (no currency conversion)"))

    budget = Budget(amount_min=best["amount_min"], amount_max=best["amount_max"],
                    currency=code, basis=basis, band=band)
    return budget, [_ev("budget", text, best["start"], best["end"])], flags


# ---------------------------------------------------------------------------
# Whole profile
# ---------------------------------------------------------------------------


def _missing_flags(p: ClientProfile) -> list[Flag]:
    flags = []
    if p.language is None:
        flags.append(Flag(field="language", kind="missing", message="language could not be detected"))
    if not p.destination:
        flags.append(Flag(field="destination", kind="missing", message="no destination found"))
    d = p.dates
    if d.start is None and d.duration is None and not d.ambiguous:
        flags.append(Flag(field="dates", kind="missing", message="no travel dates or trip length found"))
    if p.travellers.party_size is None:
        flags.append(Flag(field="travellers", kind="missing", message="number of travellers not clear"))
    if p.budget.amount_min is None and p.budget.amount_max is None:
        flags.append(Flag(field="budget", kind="missing", message="no budget found"))
    return flags


def extract_profile(text: str) -> ClientProfile:
    evidence: list[Evidence] = []
    flags: list[Flag] = []

    destination, ev = _match_lexicon(text, _DESTINATIONS, "destination")
    evidence += ev
    accommodation, ev = _match_lexicon(text, _ACCOMMODATION, "accommodation")
    evidence += ev
    flights, ev = extract_flights(text)
    evidence += ev
    interests, ev = _match_lexicon(text, _INTERESTS, "interests")
    evidence += ev
    dietary, ev = _match_lexicon(text, _DIETARY, "dietary")
    evidence += ev
    occasion, ev = _match_lexicon(text, _OCCASIONS, "occasion")
    evidence += ev
    special, ev = _match_lexicon(text, _SPECIAL, "special_requests")
    evidence += ev

    date_result = parse_dates(text)
    evidence += date_result.evidence
    flags += date_result.flags

    travellers, ev, fl = extract_travellers(text)
    evidence += ev
    flags += fl

    budget, ev, fl = extract_budget(text)
    evidence += ev
    flags += fl

    profile = ClientProfile(
        language=detect_language(text),
        destination=destination,
        dates=date_result.dates,
        travellers=travellers,
        budget=budget,
        accommodation=accommodation,
        flights=flights,
        interests=interests,
        dietary=dietary,
        occasion=occasion,
        special_requests=special,
        evidence=sorted(evidence, key=lambda e: (e.start, e.field)),
        flags=flags,
    )
    profile.flags = _missing_flags(profile) + flags
    return profile
