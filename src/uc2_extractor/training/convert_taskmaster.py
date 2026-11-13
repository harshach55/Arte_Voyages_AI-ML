"""Convert Taskmaster-2 hotels and flights into training examples. Training data only.

Only USER turns are used, because they are the closest thing to a client speaking. A slot
is kept only when its value clearly fits our label; every other annotated piece of text
goes into "ignore" so the model is told nothing about it. See docs/uc2_label_scheme.md
for the full table.

Each dialogue turn becomes one example. Turns with no annotation at all are kept only as a
small sample of plain examples (NEGATIVE_SHARE of the labelled count).

Usage:
    python -m src.uc2_extractor.training.convert_taskmaster
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

from .common import RAW_DIR, TRAINING_DIR, finish, ignored, label_counts, on_token_boundaries, span, trim, write_jsonl
from .dedupe import build_index

NEGATIVE_SHARE = 0.15
SEED = 20261106
DIRECT = {"hotel_search.location.hotel": "DESTINATION", "flight_search.destination1": "DESTINATION",
          "flight_search.destination2": "DESTINATION", "flight_search.origin": "ORIGIN",
          "flight_search.date.depart_origin": "DATE", "flight_search.date.depart_intermediate": "DATE"}

_NUM = r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_WORDS = {w: i for i, w in enumerate("one two three four five six seven eight nine ten eleven twelve".split(), 1)}
_LENGTH = re.compile(rf"^(?:for\s+)?(?:about\s+)?({_NUM}|a)\s+(night|day|week)s?$", re.I)
_LENGTH_WORD = re.compile(r"\b(nights?|days?|weeks?)\b", re.I)
_RELATIVE_RETURN = re.compile(r"^(return|returning|back|in\b)|\b(days?|nights?|weeks?)\b", re.I)
_DATE_EXTRA = re.compile(r"\b(leave|leaving|arrive|arriving|check|checking)\b", re.I)
_PEOPLE = re.compile(rf"^(?:just\s+|only\s+)?({_NUM})\s+(?:people|persons?|guests?|travell?ers|passengers?)$", re.I)
_OF_US = re.compile(rf"^(?:the\s+)?({_NUM})\s+of\s+us$", re.I)
_ADULTS = re.compile(rf"^({_NUM})\s+adults?$", re.I)
_ADULTS_KIDS = re.compile(rf"^({_NUM}\s+adults?)(?:\s*,\s*|\s+and\s+|\s+)({_NUM}\s+(?:children|child|kids?))$", re.I)
_ME_ONLY = re.compile(r"^(?:just\s+|only\s+)(?:me|myself)$", re.I)
_COUPLE = re.compile(r"^(?:me and my|myself and my|i and my)\s+(?:wife|husband|girlfriend|boyfriend|partner)$"
                     r"|^my\s+(?:wife|husband|girlfriend|boyfriend|partner)\s+and\s+i$", re.I)
_STAR = re.compile(r"\bstars?\b|-star\b", re.I)
_CABIN = re.compile(
    r"^(?:an?\s+)?(business(?:[- ]class)?|first[- ]class|economy(?:[- ]class)?|coach(?:[- ]class)?|premium economy)"
    r"(?:\s+seat(?:s|ing)?)?$", re.I)
_DIRECT_FLIGHT = re.compile(r"^(?:an?\s+)?(?:non[- ]?stops?|direct|no\s+layovers?|no\s+stops)(?:\s+flights?)?$", re.I)


def _number(text: str) -> int:
    token = re.match(rf"\s*({_NUM})", text, re.I).group(1).lower()
    return int(token) if token.isdigit() else _WORDS[token]


def _cabin_value(text: str) -> str:
    low = text.lower()
    if "business" in low:
        return "business class"
    if "first" in low:
        return "first class"
    if "premium" in low:
        return "premium economy"
    return "economy class"


def classify(name: str, value: str) -> Optional[list[tuple[str, int, int, object]]]:
    """Map one annotation to (label, start, end, canonical value) parts of the text, or None to ignore it."""
    name = re.sub(r"\s+", "", name)
    whole = len(value)
    if name in DIRECT:
        return [(DIRECT[name], 0, whole, None)]
    if name in ("hotel_search.check-in_date", "hotel_search.check-out_date"):
        return None if _DATE_EXTRA.search(value) else [("DATE", 0, whole, None)]
    if name == "flight_search.date.return":
        return None if _RELATIVE_RETURN.search(value) else [("DATE", 0, whole, None)]
    if name == "hotel_search.date_range":
        m = _LENGTH.match(value)
        if m:
            amount = 1 if m.group(1).lower() == "a" else _number(m.group(1))
            return [("DURATION", 0, whole, f"{amount} {m.group(2).lower()}s")]
        return None if _LENGTH_WORD.search(value) else [("DATE", 0, whole, None)]
    if name in ("hotel_search.num.guests", "flight_search.num.pax"):
        if _PEOPLE.match(value) or _OF_US.match(value):
            return [("PARTY", 0, whole, _number(_strip_just(value)))]
        if _ME_ONLY.match(value):
            return [("PARTY", 0, whole, 1)]
        if _ADULTS.match(value):
            return [("ADULTS", 0, whole, _number(value))]
        if _COUPLE.match(value):
            return [("ADULTS", 0, whole, 2)]
        m = _ADULTS_KIDS.match(value)
        if m:
            return [("ADULTS", m.start(1), m.end(1), _number(m.group(1))),
                    ("CHILDREN", m.start(2), m.end(2), _number(m.group(2)))]
        return None
    if name == "hotel_search.star_rating":
        return [("ACCOMMODATION", 0, whole, None)] if _STAR.search(value) else None
    if name == "flight_search.seating_class":
        return [("FLIGHT", 0, whole, _cabin_value(value))] if _CABIN.match(value) else None
    if name == "flight_search.stops":
        return [("FLIGHT", 0, whole, "direct flights")] if _DIRECT_FLIGHT.match(value) else None
    return None


def _strip_just(value: str) -> str:
    return re.sub(r"^(?:just|only|the)\s+", "", value, flags=re.I)


def convert_turn(text: str, segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (spans, ignore) for one USER turn."""
    candidates = []
    ignore_ranges = []
    for seg in segments:
        start, end = seg["start_index"], seg["end_index"]
        names = [a["name"] for a in seg.get("annotations", [])]
        results = [classify(n, text[start:end]) for n in names]
        if not results or any(r is None for r in results):
            ignore_ranges.append((start, end))
            continue
        labels = {tuple((label, s, e) for label, s, e, _ in r) for r in results}
        if len(labels) != 1:  # two annotations disagree
            ignore_ranges.append((start, end))
            continue
        for label, s, e, value in results[0]:
            a, b = trim(text, start + s, start + e)
            if a < b and on_token_boundaries(text, a, b):
                candidates.append((a, b, label, value))
            else:
                ignore_ranges.append((start, end))

    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    spans: list[dict] = []
    for a, b, label, value in candidates:
        clash = next((s for s in spans if a < s["end"] and s["start"] < b), None)
        if clash is None:
            spans.append(span(text, a, b, label, value))
        elif clash["label"] != label:  # overlapping labels that disagree: trust neither
            spans.remove(clash)
            ignore_ranges += [(clash["start"], clash["end"]), (a, b)]

    ignore = []
    for start, end in sorted(set(ignore_ranges)):
        a, b = trim(text, start, end)
        if a < b and not any(a < s["end"] and s["start"] < b for s in spans):
            ignore.append(ignored(text, a, b))
    return spans, ignore


def convert(data_dir: Path, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    labelled, plain = [], []
    for domain in ("hotels", "flights"):
        with open(data_dir / f"{domain}.json", encoding="utf-8") as f:
            dialogs = json.load(f)
        for dialog in dialogs:
            for turn in dialog["utterances"]:
                if turn["speaker"] != "USER" or not turn.get("text", "").strip():
                    continue
                example = {"id": f"tm2-{domain}-{dialog['conversation_id']}-{turn['index']}", "source": "taskmaster2",
                           "language": "en", "text": turn["text"], "spans": [], "ignore": [], "meta": {"domain": domain}}
                segments = turn.get("segments", [])
                if segments:
                    example["spans"], example["ignore"] = convert_turn(turn["text"], segments)
                    if example["spans"]:
                        labelled.append(example)
                elif len(turn["text"].split()) >= 4:
                    plain.append(example)
    rng.shuffle(plain)
    return labelled + plain[:int(len(labelled) * NEGATIVE_SHARE)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR / "taskmaster2" / "data")
    parser.add_argument("--out", type=Path, default=TRAINING_DIR / "converted" / "taskmaster2.jsonl")
    args = parser.parse_args(argv)
    index, protected = build_index()
    kept, dropped = finish(convert(args.raw), index)
    write_jsonl(args.out, kept)
    report = {"kept": len(kept), "with_spans": sum(1 for e in kept if e["spans"]), "dropped": dict(dropped),
              "protected_texts": protected, "labels": label_counts(kept)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
