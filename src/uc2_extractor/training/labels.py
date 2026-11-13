"""BIO label scheme for the slot-filling model, and the checks every example must pass.

The scheme is described in docs/uc2_label_scheme.md. Training examples keep their
labels as character spans, so they stay independent of any model's tokenizer. BIO tags
over the project tokenizer (text.tokenize) are derived from the spans, and the round
trip spans -> BIO -> spans is part of validation.

One example is a dict:
    {"id": "...", "source": "...", "language": "en", "text": "...",
     "spans": [{"start": 0, "end": 5, "label": "DESTINATION", "text": "Nepal"}],
     "ignore": [{"start": 9, "end": 20, "text": "$300 a night"}]}

"ignore" lists text that a source annotated with something that does not map cleanly
onto our labels, such as a nightly hotel price. Its tokens get the tag IGNORE_TAG and a
trainer leaves them out of the loss, so the model is told nothing about them instead of
being taught that they are plain text.
"""

from __future__ import annotations

from ..text import tokenize

LABELS = (
    "DESTINATION",      # a place the client wants to visit
    "ORIGIN",           # the city or airport they would fly from
    "DATE",             # a travel date, month or date range, as written
    "DURATION",         # the length of the whole trip
    "PARTY",            # the total number of travellers
    "ADULTS",           # the number of adults
    "CHILDREN",         # the number of children
    "AGE",              # ages of travellers
    "BUDGET",           # the money for the trip, with its qualifier, currency and basis
    "ACCOMMODATION",    # the kind of place to stay
    "FLIGHT",           # flight wishes: cabin, direct, domestic, or no flights needed
    "INTEREST",         # what they want to do or see
    "DIETARY",          # food needs
    "OCCASION",         # what the trip celebrates
    "SPECIAL_REQUEST",  # services and must-haves
)
LANGUAGES = ("en", "fr", "de")
IGNORE_TAG = "IGN"

TAGS = ("O",) + tuple(f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I"))
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}


class ExampleError(ValueError):
    """An example failed validation."""


def spans_to_bio(text: str, spans: list[dict], ignore: list[dict] | None = None) -> tuple[list[str], list[str]]:
    """Return (tokens, tags). Raises ExampleError if a span does not sit on token boundaries."""
    tokens = tokenize(text)
    tags = ["O"] * len(tokens)
    for span in sorted(spans, key=lambda s: s["start"]):
        inside = [i for i, t in enumerate(tokens) if t.start >= span["start"] and t.end <= span["end"]]
        if not inside:
            raise ExampleError(f"span {span['text']!r} covers no whole token")
        if tokens[inside[0]].start != span["start"] or tokens[inside[-1]].end != span["end"]:
            raise ExampleError(f"span {span['text']!r} does not sit on token boundaries")
        for k, i in enumerate(inside):
            if tags[i] != "O":
                raise ExampleError(f"span {span['text']!r} overlaps another span")
            tags[i] = ("B-" if k == 0 else "I-") + span["label"]
    for part in ignore or []:
        for i, t in enumerate(tokens):
            if t.start < part["end"] and part["start"] < t.end and tags[i] == "O":
                tags[i] = IGNORE_TAG
    return [t.text for t in tokens], tags


def bio_to_spans(text: str, tags: list[str]) -> list[dict]:
    """Rebuild character spans from BIO tags over the project tokenizer."""
    tokens = tokenize(text)
    if len(tokens) != len(tags):
        raise ExampleError("tags and tokens differ in length")
    spans: list[dict] = []
    current = None
    for token, tag in zip(tokens, tags):
        prefix, _, label = tag.partition("-")
        if prefix == "B" or (prefix == "I" and (current is None or current["label"] != label)):
            current = {"start": token.start, "end": token.end, "label": label}
            spans.append(current)
        elif prefix == "I":
            current["end"] = token.end
        else:
            current = None
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]
    return spans


def _check_position(text: str, part: dict) -> None:
    if not 0 <= part["start"] < part["end"] <= len(text):
        raise ExampleError(f"span {part['start']}-{part['end']} is outside the text")
    if text[part["start"]:part["end"]] != part["text"]:
        raise ExampleError(f"span text {part['text']!r} does not match the enquiry at {part['start']}-{part['end']}")
    if part["text"] != part["text"].strip():
        raise ExampleError(f"span {part['text']!r} has spaces at its edge")


def validate_example(example: dict) -> None:
    """Raise ExampleError unless the example is exactly right."""
    for key in ("id", "source", "language", "text", "spans"):
        if key not in example:
            raise ExampleError(f"missing key {key}")
    text = example["text"]
    if not text.strip():
        raise ExampleError("empty text")
    if example["language"] not in LANGUAGES:
        raise ExampleError(f"unknown language {example['language']!r}")
    spans = sorted(example["spans"], key=lambda s: s["start"])
    ignore = example.get("ignore", [])
    last_end = -1
    for span in spans:
        if span["label"] not in LABELS:
            raise ExampleError(f"unknown label {span['label']!r}")
        _check_position(text, span)
        if span["start"] < last_end:
            raise ExampleError(f"span {span['text']!r} overlaps the span before it")
        last_end = span["end"]
    for part in ignore:
        _check_position(text, part)
        if any(part["start"] < s["end"] and s["start"] < part["end"] for s in spans):
            raise ExampleError(f"ignored text {part['text']!r} overlaps a labelled span")
    _, tags = spans_to_bio(text, spans, ignore)
    rebuilt = bio_to_spans(text, tags)
    wanted = sorted((s["start"], s["end"], s["label"]) for s in spans)
    if sorted((s["start"], s["end"], s["label"]) for s in rebuilt) != wanted:
        raise ExampleError("spans do not survive the BIO round trip")
