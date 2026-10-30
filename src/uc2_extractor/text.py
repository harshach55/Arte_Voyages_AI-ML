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
    },
    "fr": {
        "le", "la", "les", "et", "nous", "voudrions", "pour", "avec", "un", "une",
        "des", "du", "est", "sommes", "notre", "nos", "je", "mon", "ma", "mes",
        "souhaitons", "au", "aux", "vous", "merci", "voyage", "sur", "dans", "pas",
        "être", "avons", "aimerions", "enfants", "qui",
    },
    "de": {
        "der", "das", "und", "wir", "möchten", "für", "mit", "ein", "eine", "einen",
        "ist", "sind", "unser", "unsere", "ich", "mein", "meine", "nach", "im", "zu",
        "von", "auf", "bitte", "reise", "gerne", "uns", "dem", "den", "nicht", "auch",
        "würden", "haben", "kinder", "wäre",
    },
}


def detect_language(text: str, min_hits: int = 2) -> Optional[str]:
    """Return 'en', 'fr' or 'de', or None when the text gives too little signal.

    Counts stopwords per language. Returns None if the best language has fewer
    than min_hits stopwords or ties with another language.
    """
    words = [t.text.lower() for t in tokenize(text)]
    scores = {lang: sum(w in stops for w in words) for lang, stops in STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score < min_hits or best_score == second_score:
        return None
    return best
