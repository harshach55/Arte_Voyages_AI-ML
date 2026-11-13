"""Keep training data apart from test data.

ProtectedIndex holds hashes of the gold set, held-out sets 1 and 2 and the corrections, never
their text. check() says why a candidate is too close to any of them, or returns None.

Rules (all after normalising: lower case, no accents, punctuation removed):
- exact:         the whole text equals a test text
- shared_run:    they share a run of RUN_WORDS or more consecutive words
- near_text:     SHINGLE_CONTAINMENT or more of the shorter text's 3-word groups are in a test text
- near_chars:    the 5-letter groups overlap by CHAR_JACCARD or more (catches typo variants)
- near_sentence: a sentence of 6+ words overlaps a test sentence by SENTENCE_JACCARD or more
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "uc2_inquiries"

RUN_WORDS = 6
SHINGLE_CONTAINMENT = 0.5
CHAR_JACCARD = 0.6
SENTENCE_JACCARD = 0.5
MIN_SENTENCE_WORDS = 6

_SENTENCE = re.compile(r"[.!?;:\n]+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _hash(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()


def _word_groups(words: list[str], n: int) -> set[bytes]:
    return {_hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def _char_groups(norm: str, n: int = 5) -> set[bytes]:
    return {_hash(norm[i:i + n]) for i in range(len(norm) - n + 1)}


def _sentences(text: str) -> list[list[str]]:
    parts = (normalise(s).split() for s in _SENTENCE.split(text))
    return [p for p in parts if len(p) >= MIN_SENTENCE_WORDS]


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class ProtectedIndex:
    """Hashes of protected text. Holds no readable text, so it is safe to print or log results."""

    def __init__(self) -> None:
        self.exact: set[bytes] = set()
        self.runs: set[bytes] = set()
        self.docs: list[tuple[set[bytes], set[bytes]]] = []  # (word 3-groups, char 5-groups)
        self.sentences: list[set[bytes]] = []
        self.count = 0

    def add(self, text: str) -> None:
        norm = normalise(text)
        if not norm:
            return
        words = norm.split()
        self.exact.add(_hash(norm))
        self.runs |= _word_groups(words, RUN_WORDS)
        self.docs.append((_word_groups(words, 3), _char_groups(norm)))
        self.sentences += [_word_groups(s, 2) for s in _sentences(text)]
        self.count += 1

    def check(self, text: str) -> Optional[str]:
        """Return the reason this text is too close to protected text, or None."""
        norm = normalise(text)
        words = norm.split()
        if _hash(norm) in self.exact:
            return "exact"
        if _word_groups(words, RUN_WORDS) & self.runs:
            return "shared_run"
        groups, chars = _word_groups(words, 3), _char_groups(norm)
        for doc_groups, doc_chars in self.docs:
            if groups and doc_groups:
                if len(groups & doc_groups) / min(len(groups), len(doc_groups)) >= SHINGLE_CONTAINMENT:
                    return "near_text"
            smaller, larger = sorted((len(chars), len(doc_chars)))
            if larger and smaller / larger >= CHAR_JACCARD and _jaccard(chars, doc_chars) >= CHAR_JACCARD:
                return "near_chars"
        for sentence in _sentences(text):
            s = _word_groups(sentence, 2)
            if any(_jaccard(s, other) >= SENTENCE_JACCARD for other in self.sentences):
                return "near_sentence"
        return None


def protected_texts(data_dir: Path = DATA) -> Iterable[tuple[str, str]]:
    """Yield (set name, text) for every enquiry that must stay out of training data."""
    for name in ("gold", "heldout", "heldout2"):
        path = data_dir / name / "enquiries.csv"
        if path.exists():
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("text", "").strip():
                        yield name, row["text"]
    corrections = data_dir / "corrections" / "corrections.jsonl"
    if corrections.exists():
        with open(corrections, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield "corrections", json.loads(line)["text"]


def build_index(data_dir: Path = DATA) -> tuple[ProtectedIndex, dict[str, int]]:
    """Build the index and return it with the number of texts per protected set."""
    index = ProtectedIndex()
    counts: dict[str, int] = {}
    for name, text in protected_texts(data_dir):
        index.add(text)
        counts[name] = counts.get(name, 0) + 1
    return index, counts
