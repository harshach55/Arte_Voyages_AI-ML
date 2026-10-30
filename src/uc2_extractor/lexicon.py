"""Word lists for English, French and German used by the rule-based extractors.

All keys are lower case. Matching is case-insensitive.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------

MONTHS = {
    # English (May is handled separately because it is also a verb)
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # French
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    # German
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "maerz": 3, "juni": 6,
    "juli": 7, "oktober": 10, "dezember": 12,
}

# ---------------------------------------------------------------------------
# Small numbers written as words
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fourteen": 14, "fifteen": 15, "twenty": 20,
    # French ("six" is spelled as in English; "sept" is also an English
    # month abbreviation, the surrounding pattern decides)
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "quatorze": 14, "quinze": 15, "vingt": 20,
    # German
    "ein": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1, "zwei": 2,
    "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8,
    "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12, "vierzehn": 14,
    "fünfzehn": 15, "zwanzig": 20,
}


def alternation(words) -> str:
    """Regex alternation of words, longest first so 'sept' loses to 'september'."""
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


def to_number(token: str) -> int | None:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)
