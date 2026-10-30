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


# destinations also accept a German genitive "s" (Sri Lankas, Indiens)
_DESTINATIONS = {
    label: re.compile(rf"(?<!\w)(?:{'|'.join(frags)})s?(?!\w)", _I)
    for label, frags in lx.DESTINATIONS.items()
}
# "a family from Mumbai", "wir kommen aus Delhi": the client's home, not a destination
_ORIGIN_BEFORE = re.compile(
    r"\b(?:from|out of|depuis|au d[ée]part de|arriv\w* de|venons de|aus|ab)\s+$", _I
)
_ROUTE_AFTER = re.compile(r"^\s*(?:to|à|nach|bis|then)\b", _I)
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


def _lexicon_hits(text: str, compiled: dict[str, re.Pattern], taken=None) -> list[tuple[int, int, str]]:
    """Find (start, end, label) lexicon hits. Longer matches win over shorter
    overlapping ones (e.g. 'water villa' beats 'villa'). Negated matches are
    dropped. Spans in `taken` (already used by another field) are skipped."""
    hits = [
        (m.start(), m.end(), label)
        for label, rx in compiled.items()
        for m in rx.finditer(text)
    ]
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    taken = list(taken or [])
    kept = []
    for s, e, label in hits:
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        if not _negated(text, s):
            kept.append((s, e, label))
    return sorted(kept)


def _to_field(text: str, hits, field: str):
    labels = list(dict.fromkeys(label for _, _, label in hits))
    return labels, [_ev(field, text, s, e) for s, e, _ in hits]


def _match_lexicon(text: str, compiled: dict[str, re.Pattern], field: str, taken=None):
    return _to_field(text, _lexicon_hits(text, compiled, taken), field)


_BARE_ANNIVERSAIRE = re.compile(r"anniversaire", _I)
_MARRIAGE_WORDS = re.compile(r"\b(?:mariage|noces)\b", _I)


def extract_occasions(text: str):
    hits = _lexicon_hits(text, _OCCASIONS)
    # in French "anniversaire" alone means birthday, unless the enquiry is
    # about a wedding anniversary ("nos 40 ans de mariage ... la soirée d'anniversaire")
    if _MARRIAGE_WORDS.search(text):
        hits = [
            (s, e, label) for s, e, label in hits
            if not (label == "birthday" and _BARE_ANNIVERSAIRE.fullmatch(text[s:e]))
        ]
    return _to_field(text, hits, "occasion")


def extract_destinations(text: str):
    hits = [
        (s, e, label) for s, e, label in _lexicon_hits(text, _DESTINATIONS)
        if not (_ORIGIN_BEFORE.search(text[max(0, s - 25):s]) and not _ROUTE_AFTER.search(text[e:e + 10]))
    ]
    return _to_field(text, hits, "destination")


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


# "from London and Frankfurt": a further city joined to a departure city
_NEXT_CITY = {
    city: re.compile(rf"\s*(?:{lx.ORIGIN_JOINERS})\s*(?P<city>{'|'.join(frags)})(?!\w)", _I)
    for city, frags in lx.FLIGHT_ORIGINS.items()
}


def _next_city(text: str, pos: int, end: int):
    for city, rx in _NEXT_CITY.items():
        m = rx.match(text, pos, end)
        if m:
            return city, m
    return None


def extract_flights(text: str):
    labels, evidence = _match_lexicon(text, _FLIGHT_CLASSES, "flights")

    def add(city, s, e):
        label = f"departure from {city}"
        if label not in labels:
            labels.append(label)
        evidence.append(_ev("flights", text, s, e))

    # a departure city only counts in a sentence that talks about flying
    for sent in _SENTENCE_RE.finditer(text):
        if not _FLIGHT_WORD_RE.search(sent.group()):
            continue
        for city, rx in _ORIGINS.items():
            for m in rx.finditer(text, sent.start(), sent.end()):
                if _negated(text, m.start()):
                    continue
                add(city, m.start(), m.end())
                nxt = _next_city(text, m.end(), sent.end())
                while nxt:
                    next_city, n = nxt
                    add(next_city, n.start("city"), n.end("city"))
                    nxt = _next_city(text, n.end(), sent.end())
    return labels, sorted(evidence, key=lambda x: x.start)


# ---------------------------------------------------------------------------
# Travellers
# ---------------------------------------------------------------------------

_NUMW = lx.alternation(lx.NUMBER_WORDS)
_N = rf"(?P<n>\d{{1,2}}|{_NUMW})"
# after "we are" / "nous sommes" / "wir sind", "un/une/ein/eine" is an article
# ("wir sind eine Familie"), so only numbers from two upwards count
_NUMW2 = lx.alternation(w for w, v in lx.NUMBER_WORDS.items() if v >= 2)
_N2 = rf"(?P<n>\d{{1,2}}|{_NUMW2})"
# German "zu dritt" (three of us) etc.
_GROUP_WORDS = {"dritt": 3, "viert": 4, "fünft": 5, "funft": 5, "sechst": 6, "siebt": 7, "acht": 8}
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
# (but not "a baby cot" or "baby-sitting")
_ONE_CHILD_RE = re.compile(
    r"(?<!\w)(?:a|an|our|un|une|notre|ein(?:em|en)?|unser(?:em|en)?)\s+"
    r"(?:baby|b[ée]b[ée]|infant|nourrisson|kleinkind|son|daughter|child|fils|fille|enfant|sohn|tochter|kind)(?!\w)"
    r"(?![\s-]*(?:cot|crib|sitt\w*|bett|club|food|seat))",
    _I,
)
# (pattern, kind): kind "n" gives the party size, "n_adults" gives party size
# and adults, "group" uses the German "zu dritt" words
_TOTAL_RES = [
    (re.compile(rf"(?:party|group|groupe|gruppe|family|famille|familie) (?:of|de|von) {_N}(?!\w)", _I), "n"),
    (re.compile(
        rf"(?<!\w){_N}\s+(?:people|persons|guests|travell?ers|pax|personnes|voyageurs|personen|"
        rf"reisende[n]?|g[äa]ste[n]?|teilnehmer)(?!\w)", _I), "n"),
    (re.compile(
        rf"(?:we are|we're|we will be|we'll be|there are|there will be|nous sommes|nous serons|nous serions|"
        rf"on est|on sera|wir sind|wir werden|wir w[äa]ren)\s+{_N2}(?!\w){_UNIT_AFTER}", _I), "n"),
    (re.compile(rf"(?<!\w){_N}k[öo]pfige?\w*", _I), "n"),
    (re.compile(
        rf"(?<!\w){_N2}\s+(?:friends|colleagues|amis|amies|copains|copines|potes|coll[èe]gues|"
        rf"freunde|freundinnen|kollegen|kolleginnen)(?!\w)", _I), "n_adults"),
    (re.compile(r"\bzu\s+(?P<g>dritt|viert|f[üu]nft|sechst|siebt|acht)(?!\w)", _I), "group"),
    (re.compile(rf"\bnous\s+{_N2}(?!\w){_UNIT_AFTER}", _I), "n"),
    (re.compile(rf"\b(?:for|f[üu]r|pour)\s+(?P<n>{_NUMW2})(?!\w){_UNIT_AFTER}(?!\s+(?:of|de|von)\b)", _I), "n"),
]
_COUPLE_RE = re.compile(
    r"(?:my|our)\s+(?:wife|husband|partner|fianc[ée]e?|girlfriend|boyfriend|spouse)\s+and\s+(?:I|me)(?!\w)"
    r"|me and my\s+(?:wife|husband|partner|fianc[ée]e?|girlfriend|boyfriend|bf|gf|spouse)(?!\w)"
    r"|(?:ma femme|mon mari|mon [ée]pouse|mon [ée]poux|mon compagnon|ma compagne|mon partenaire|"
    r"ma partenaire|mon conjoint|ma conjointe|mon fianc[ée]|ma fianc[ée]e)\s+et\s+moi"
    r"|(?:meine (?:ehe)?frau|mein (?:ehe)?mann|mein partner|meine partnerin|meine freundin|mein freund|"
    r"mein verlobter|meine verlobte)\s+und\s+ich"
    r"|as a couple|(?:the )?two of us|(?:a|retired|young|married|elderly) couple(?! of)|en couple|un couple"
    r"|nous deux|tous les deux|zu zweit|wir beide|uns beide|ehepaar|als paar",
    _I,
)
_SOLO_RE = re.compile(
    r"travell?ing (?:alone|solo)|solo travell?er|on my own|by myself|voyage(?:r|rai)? seule?"
    r"|seule? voyageu(?:r|se)|allein(?:e)? reisen|alleinreisende?r?|reise allein",
    _I,
)
_NUMLIST = r"\d{1,2}(?:\s*(?:,|and|&|et|und)\s*\d{1,2})*"
_NOT_MONTHS = r"(?!\d)(?!\s*(?:months?|mois|monate))"
_CHILD_NOUN = (
    r"children|child|kids?|enfants?|kinder|teenagers?|teens?|adolescents?|sons?|daughters?|"
    r"fils|filles?|s[öo]hne|sohn|t[öo]chter|tochter|boys?|girls?|kind"
)
# (pattern, kind): kind "list" reads all numbers, "both" doubles one age,
# "months" means a baby under one year (age 0)
_AGE_RES = [
    (re.compile(rf"(?:[âa]g[ée]e?s?)\s+de\s+(?P<list>{_NUMLIST})\s+ans(?!\w)", _I), "list"),
    (re.compile(rf"\b(?:aged?|ages)\s+(?P<list>{_NUMLIST})", _I), "list"),
    (re.compile(rf"(?P<list>{_NUMLIST})\s*(?:-|\s)years?(?:-|\s)olds?(?!\w)", _I), "list"),
    (re.compile(rf"im alter von\s+(?P<list>{_NUMLIST})", _I), "list"),
    (re.compile(rf"(?P<list>{_NUMLIST})\s+jahre(?:n)?\s+alt(?!\w)", _I), "list"),
    (re.compile(r"(?<!\d)(?P<list>\d{1,2})-?j[äa]hrig\w*", _I), "list"),
    (re.compile(rf"\bmit\s+(?P<list>\d{{1,2}})\s+jahren(?!\w)", _I), "list"),
    # "3 kids (4, 7, 11)", "2 Kinder: 6 und 9"
    (re.compile(rf"(?:{_CHILD_NOUN})\s*(?:\(|:)\s*(?P<list>{_NUMLIST}){_NOT_MONTHS}", _I), "list"),
    # "1 teenager of 15"
    (re.compile(rf"(?:{_CHILD_NOUN})\s+(?:of|von)\s+(?P<list>\d{{1,2}}){_NOT_MONTHS}", _I), "list"),
    # "solo traveller (62)", "ich (34)", "allein (62 Jahre)"
    (re.compile(
        r"(?:travell?er|ich|i'm|i am|je|seule?|allein|myself)\s*\(\s*(?P<list>\d{1,2})"
        r"(?:\s*(?:jahre|ans|years?(?: old)?))?\s*\)", _I), "list"),
    (re.compile(r"(?:we're|we are|wir sind|nous avons)\s+(?:both|beide|tous (?:les )?deux)\s+(?P<list>\d{1,2})(?!\d)", _I), "both"),
    (re.compile(r"\(\s*\d{1,2}\s*(?:months?|mois|monate?n?)(?:\s*old)?\s*\)|\d{1,2}[- ]months?[- ]old", _I), "months"),
    (re.compile(rf"(?<!\d)(?P<list>{_NUMLIST})\s+ans(?!\w)(?!\s+de\s+mariage)", _I), "list"),
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
    for rx, kind in _TOTAL_RES:
        m = rx.search(text)
        if m:
            if kind == "group":
                total = _GROUP_WORDS[m.group("g").lower()]
            else:
                total = _num(m.group("n"))
            if kind == "n_adults" and adults is None:
                adults = total
            evidence.append(_ev("travellers", text, m.start(), m.end()))
            break

    found_ages: list[tuple[int, list[int]]] = []
    taken: list[tuple[int, int]] = []
    for rx, kind in _AGE_RES:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            if _NOT_AGE_BEFORE.search(text[max(0, m.start() - 12):m.start()]):
                continue
            taken.append((m.start(), m.end()))
            evidence.append(_ev("travellers", text, m.start(), m.end()))
            if kind == "months":
                values = [0]
            else:
                values = [int(x) for x in re.findall(r"\d{1,2}", m.group("list"))]
                if kind == "both":
                    values = values * 2
            found_ages.append((m.start(), values))
    ages = [a for _, values in sorted(found_ages) for a in values]

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
# Indian grouping first (4,00,000), then 20,000 / 20.000 / 20 000, then plain numbers
_AMOUNT = (
    r"\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:[   ,.'’]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
)
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
    r"(?:up to|max(?:imum)?\.?|no more than|not more than|under|below|less than|jusqu'[àa]|pas plus de|"
    r"moins de|au maximum|bis zu|bis|h[öo]chstens|maximal|unter|weniger als)\s*$",
    _I,
)
# "5k USD max", "2000 € chacun max"
_UP_TO_AFTER = re.compile(
    r"^\s*(?:(?:each|chacun\w*|pro person|par personne)\s+)?(?:max(?:imum)?\.?|tops|at most|au maximum|h[öo]chstens)(?!\w)",
    _I,
)
_AT_LEAST = re.compile(
    r"(?:at least|minimum|from|starting at|au moins|[àa] partir de|mindestens|ab)\s*$", _I
)
_RATE_AFTER = re.compile(r"^\s*(?:per|a|an|par|pro|la|die)\s+(?:night|nuit|nacht|day|jour|tag)\b", _I)
_PER_PERSON = re.compile(
    r"\b(?:per person|per head|pp|p\.\s?p\.|each|par personne|par pers\.?|chacune?|pro person|pro kopf)(?!\w)", _I
)
_PER_TRIP = re.compile(
    r"\b(?:in total|total|for everything|for (?:the )?(?:whole|entire) (?:trip|group|family)|for all of us|"
    r"for everyone|for both of us|au total|en tout|pour l'ensemble|pour (?:tout le monde|nous (?:deux|tous))|"
    r"global(?:e)?|insgesamt|gesamt|f[üu]r alles|f[üu]r alle|f[üu]r uns (?:beide|alle))(?!\w)",
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
                # "entre 20 et 25 000 €": the first number shares the thousands
                if lo < 1000 <= hi and hi % 1000 == 0 and lo * 1000 < hi:
                    lo *= 1000
                if lo >= hi:
                    continue
                amount_min, amount_max = lo, hi
            else:
                value = _parse_amount(g["a1"] or g["a2"])
                amount_min = amount_max = value
                before = text[max(0, m.start() - 25):m.start()]
                if _UP_TO.search(before) or _UP_TO_AFTER.search(text[m.end():m.end() + 20]):
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

    destination, ev = extract_destinations(text)
    evidence += ev
    accommodation, ev = _match_lexicon(text, _ACCOMMODATION, "accommodation")
    evidence += ev
    # "beachfront villa" / "am Strand" describe the hotel, not a beach interest
    accommodation_spans = [(e.start, e.end) for e in ev]
    flights, ev = extract_flights(text)
    evidence += ev
    interests, ev = _match_lexicon(text, _INTERESTS, "interests", taken=accommodation_spans)
    evidence += ev
    dietary, ev = _match_lexicon(text, _DIETARY, "dietary")
    evidence += ev
    occasion, ev = extract_occasions(text)
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
