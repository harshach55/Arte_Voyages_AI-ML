"""Convert MASSIVE (en-US, fr-FR, de-DE) into training examples. Training data only.

Only one slot maps cleanly onto our labels: date -> DATE. The mapping is deliberately
narrow, because MASSIVE is a voice-assistant dataset:
- rows come only from planning scenarios (calendar, transport, recommendation, weather)
- past dates, clock times and stray words are not used
- any one date value is used at most MAX_PER_VALUE times per language, otherwise "today"
  and "tomorrow" would swamp everything else

place_name, time and timeofday go into "ignore". place_name never says whether a place
is a destination ("weather in paris", "nach hause"), so it must not be forced into
DESTINATION, and it must not be taught as plain text either.

Usage:
    python -m src.uc2_extractor.training.convert_massive
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from .common import RAW_DIR, TRAINING_DIR, finish, ignored, label_counts, span, write_jsonl
from .dedupe import build_index

LOCALES = {"en-US": "en", "fr-FR": "fr", "de-DE": "de"}
SCENARIOS = {"calendar", "transport", "recommendation", "weather"}
MAX_PER_VALUE = 25
IGNORE_SLOTS = {"place_name", "time", "timeofday"}

_TAG = re.compile(r"\[\s*([a-z_]+)\s*:\s*(.*?)\s*\]")
_PAST = re.compile(r"\b(yesterday|yesterdays|last|ago|hier|dernier|derni[èe]re|gestern|gestrige\w*|letzte\w*)\b", re.I)
_CLOCK = re.compile(r"\b(uhr|heures?|o.clock|a\. m\.|p\. m\.|am|pm)\b", re.I)
_YEAR_WORDS = re.compile(r"(thousand|hundred|mille|cent|hundert|tausend)", re.I)
_STRAY = {"date", "datum", "mois", "month", "monats", "chaque", "maintenant", "tonight", "est", "le", "un",
          "weekly", "hebdomadaire", "daily"}


def parse_annotation(annot: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Turn 'buy [date : friday] tickets' into the plain text and (slot, start, end) offsets."""
    text = ""
    slots = []
    pos = 0
    for m in _TAG.finditer(annot):
        text += annot[pos:m.start()]
        start = len(text)
        text += m.group(2)
        slots.append((m.group(1), start, len(text)))
        pos = m.end()
    return text + annot[pos:], slots


def good_date(value: str) -> bool:
    if _PAST.search(value) or _CLOCK.search(value) or _YEAR_WORDS.search(value):
        return False
    return value.lower() not in _STRAY and len(value) > 3


def convert_row(row: dict, language: str, seen_values: Counter) -> Optional[dict]:
    """One MASSIVE row -> one example, or None if it teaches nothing clean."""
    if row["scenario"] not in SCENARIOS:
        return None
    text, slots = parse_annotation(row["annot_utt"])
    spans, ignore = [], []
    for slot, start, end in slots:
        value = text[start:end]
        if slot == "date":
            key = (language, value.lower())
            if good_date(value) and seen_values[key] < MAX_PER_VALUE:
                seen_values[key] += 1
                spans.append(span(text, start, end, "DATE"))
            else:
                ignore.append(ignored(text, start, end))
        elif slot in IGNORE_SLOTS:
            ignore.append(ignored(text, start, end))
    if not spans:
        return None
    return {"id": f"massive-{row['locale']}-{row['id']}", "source": "massive", "language": language,
            "text": text, "spans": spans, "ignore": ignore,
            "meta": {"partition": row["partition"], "scenario": row["scenario"]}}


def convert(massive_dir: Path) -> list[dict]:
    examples = []
    seen_values: Counter = Counter()
    for locale, language in LOCALES.items():
        with open(massive_dir / f"{locale}.jsonl", encoding="utf-8") as f:
            for line in f:
                example = convert_row(json.loads(line), language, seen_values)
                if example:
                    examples.append(example)
    return examples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR / "massive" / "extracted" / "1.0" / "data")
    parser.add_argument("--out", type=Path, default=TRAINING_DIR / "converted" / "massive.jsonl")
    args = parser.parse_args(argv)
    index, protected = build_index()
    kept, dropped = finish(convert(args.raw), index)
    write_jsonl(args.out, kept)
    report = {"kept": len(kept), "dropped": dict(dropped), "protected_texts": protected, "labels": label_counts(kept)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
