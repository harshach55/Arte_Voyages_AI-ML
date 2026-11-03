"""Tokenizer with character offsets, and a simple language detector.

The tokenizer never changes the text: every token keeps the exact start and
end position in the original string, so text[start:end] == token.text.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

# Order matters: numbers with separators first (20,000 / 20.000 / 3/4),
# then words (hyphens kept inside a word), then any other single character.
TOKEN_RE = re.compile(r"\d+(?:[.,:/]\d+)*|\w+(?:-\w+)*|[^\w\s]")


class Token(NamedTuple):
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# Common function words. Words shared between languages (in, die, de, a ...)
# are left out on purpose so they do not count for two languages.
STOPWORDS = {
    "en": {
        "the", "and", "we", "would", "like", "to", "our", "of", "for", "with",
        "is", "are", "my", "us", "please", "from", "trip", "looking", "travel",
        "be", "have", "will", "you", "this", "that", "at", "on", "it", "an",
        "your", "children", "could", "want",
        # travel words that only exist in English
        "two", "three", "four", "adults", "kids", "nights", "weeks", "days", "until",
        "holiday", "honeymoon", "around", "about",
    },
    "fr": {
        "le", "la", "les", "et", "nous", "voudrions", "pour", "avec", "un", "une",
        "des", "du", "est", "sommes", "notre", "nos", "je", "mon", "ma", "mes",
        "souhaitons", "au", "aux", "vous", "merci", "voyage", "sur", "dans", "pas",
        "être", "avons", "aimerions", "enfants", "qui",
        # travel words that only exist in French
        "deux", "trois", "quatre", "adultes", "nuits", "semaines", "jours", "voulons",
        "partir", "pendant", "séjour", "vacances", "environ",
    },
    "de": {
        "der", "das", "und", "wir", "möchten", "für", "mit", "ein", "eine", "einen",
        "ist", "sind", "unser", "unsere", "ich", "mein", "meine", "nach", "im", "zu",
        "von", "auf", "bitte", "reise", "gerne", "uns", "dem", "den", "nicht", "auch",
        "würden", "haben", "kinder", "wäre",
        # travel words that only exist in German
        "zwei", "drei", "vier", "erwachsene", "nächte", "wochen", "tage", "vom", "bis",
        "reisen", "urlaub", "etwa", "oder",
    },
}

# Letters that point to one language when there are too few words to count.
# Accented capitals are covered because the text is lower-cased first.
LETTER_HINTS = {
    "de": set("äöüß"),
    "fr": set("éèêàçùôîœ"),
}


def detect_language(text: str, min_hits: int = 2) -> Optional[str]:
    """Return 'en', 'fr' or 'de', or None when the text gives too little signal.

    Counts stopwords per language. If the best language has fewer than
    min_hits stopwords or ties with another language, letters such as ä/ö/ü/ß
    (German) or é/è/ç (French) can decide, but only when they point to one
    language and agree with any stopwords found. Otherwise returns None.
    """
    words = [t.text.lower() for t in tokenize(text)]
    scores = {lang: sum(w in stops for w in words) for lang, stops in STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score >= min_hits and best_score > second_score:
        return best

    lowered = text.lower()
    hinted = [lang for lang, letters in LETTER_HINTS.items() if any(ch in letters for ch in lowered)]
    if len(hinted) == 1 and scores[hinted[0]] == max(scores.values()):
        return hinted[0]
    return None
