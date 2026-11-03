"""Small spelling-tolerant matching for typos such as "Sri Lnka" or "Novmber".

Only one edit is allowed (a letter added, removed, changed, or two neighbouring
letters swapped), only for words of at least `min_len` letters, and the first
letter must match. This keeps false matches rare.
"""

from __future__ import annotations

import re
from typing import Iterable

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def edit_distance(a: str, b: str, limit: int = 1) -> int:
    """Optimal string alignment distance, stopping early above `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def close_match(word: str, forms: Iterable[str], min_len: int = 6) -> str | None:
    """Return the single form that `word` is one edit away from, else None."""
    w = word.lower()
    hits = {
        f for f in forms
        if len(f) >= min_len and f[0] == w[0] and f != w and edit_distance(w, f) <= 1
    }
    return hits.pop() if len(hits) == 1 else None


def word_spans(text: str, max_words: int = 2) -> list[tuple[int, int, str]]:
    """All runs of 1..max_words consecutive words, as (start, end, text)."""
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    spans = []
    for i in range(len(words)):
        for n in range(1, max_words + 1):
            if i + n > len(words):
                break
            s, e = words[i][0], words[i + n - 1][1]
            # only join words separated by a single space or hyphen
            if n > 1 and not re.fullmatch(r"[ -]", text[words[i + n - 2][1]:words[i + n - 1][0]]):
                break
            spans.append((s, e, text[s:e]))
    return spans
