"""Shared steps for building training data: trimming, checking and writing examples."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..text import tokenize
from .dedupe import ProtectedIndex, normalise
from .labels import ExampleError, validate_example

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "data" / "uc2_inquiries" / "training"
RAW_DIR = TRAINING_DIR / "raw"
_EDGE = " \t\n.,;:!?\"'()[]"


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Move a span's edges inwards past spaces and punctuation."""
    while start < end and text[start] in _EDGE:
        start += 1
    while end > start and text[end - 1] in _EDGE:
        end -= 1
    return start, end


def on_token_boundaries(text: str, start: int, end: int) -> bool:
    tokens = tokenize(text)
    return any(t.start == start for t in tokens) and any(t.end == end for t in tokens)


def span(text: str, start: int, end: int, label: str, value=None) -> dict:
    out = {"start": start, "end": end, "label": label, "text": text[start:end]}
    if value is not None:
        out["value"] = value
    return out


def ignored(text: str, start: int, end: int) -> dict:
    return {"start": start, "end": end, "text": text[start:end]}


def finish(examples: Iterable[dict], index: ProtectedIndex) -> tuple[list[dict], Counter]:
    """Keep the examples that pass every check, and count why the others were dropped.

    Dropped for: failing validation, being too close to protected test text, or repeating
    an earlier example exactly.
    """
    kept: list[dict] = []
    dropped: Counter = Counter()
    seen: set[str] = set()
    for example in examples:
        try:
            validate_example(example)
        except ExampleError:
            dropped["failed_validation"] += 1
            continue
        reason = index.check(example["text"])
        if reason:
            dropped[f"too_close_to_test_text:{reason}"] += 1
            continue
        key = example["language"] + " " + normalise(example["text"])
        if key in seen:
            dropped["exact_repeat"] += 1
            continue
        seen.add(key)
        kept.append(example)
    return kept, dropped


def label_counts(examples: Iterable[dict]) -> dict[str, dict[str, int]]:
    """Per language, how many spans of each label."""
    counts: dict[str, Counter] = {}
    for e in examples:
        c = counts.setdefault(e["language"], Counter())
        c["examples"] += 1
        c.update(s["label"] for s in e["spans"])
    return {lang: dict(sorted(c.items())) for lang, c in sorted(counts.items())}


def write_jsonl(path: Path, examples: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
