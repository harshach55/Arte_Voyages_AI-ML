"""Building blocks for synthetic enquiries.

Text is built from Parts. A Part is either plain text or labelled text, and every change
(typos, lower case, dropped accents) happens on Parts before the final text exists. The
character offsets are computed once at the very end, so a span can never drift away from
its text. validate_example still checks the result independently.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class Part:
    text: str
    label: Optional[str] = None
    value: Any = None  # the canonical value behind a labelled part


Fragment = list[Part]
_SLOT = re.compile(r"\{(\w+)\}")
_WORD = re.compile(r"[^\W\d_]{4,}")


def S(label: str, text: str, value: Any = None) -> Part:
    """A labelled part."""
    return Part(text, label, value)


def T(text: str) -> Part:
    """A plain part."""
    return Part(text)


def slots_in(template: str) -> set[str]:
    return set(_SLOT.findall(template))


def fill(template: str, slots: dict[str, Fragment]) -> Fragment:
    """Replace {name} placeholders in a template with fragments."""
    parts: Fragment = []
    pos = 0
    for m in _SLOT.finditer(template):
        if m.start() > pos:
            parts.append(T(template[pos:m.start()]))
        parts.extend(slots[m.group(1)])
        pos = m.end()
    if pos < len(template):
        parts.append(T(template[pos:]))
    return parts


def join(fragments: Sequence[Fragment], separator: str, last: str) -> Fragment:
    """Join fragments as a list: a, b and c."""
    out: Fragment = []
    for i, fragment in enumerate(fragments):
        if i:
            out.append(T(last if i == len(fragments) - 1 else separator))
        out.extend(fragment)
    return out


def plain(fragment: Fragment) -> Fragment:
    """The same text with its labels removed (for refused or past items)."""
    return [T(p.text) for p in fragment]


def capitalise(fragment: Fragment) -> Fragment:
    """Upper-case the first letter of a fragment."""
    if not fragment:
        return fragment
    first = fragment[0]
    if not first.text:
        return capitalise(fragment[1:])
    return [replace(first, text=first.text[0].upper() + first.text[1:])] + fragment[1:]


def tidy(fragment: Fragment) -> Fragment:
    """Drop a full stop that follows one already there, as after "p.P." at the end of a sentence."""
    out: Fragment = []
    for part in fragment:
        if out and out[-1].text.endswith(".") and part.text.startswith(".") and not part.label:
            part = replace(part, text=part.text[1:])
        if part.text:
            out.append(part)
    return out


def map_text(fragment: Fragment, fn) -> Fragment:
    return [replace(p, text=fn(p.text)) for p in fragment]


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.replace("œ", "oe").replace("Œ", "Oe"))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def german_ascii(text: str) -> str:
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        text = text.replace(a, b)
    return text


def typo(word: str, rng: random.Random) -> str:
    """One everyday typing slip. The first letter stays, as it usually does."""
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(("swap", "drop", "double"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def add_typos(fragment: Fragment, rate: float, rng: random.Random) -> Fragment:
    """Put typos in about `rate` of the words of four letters or more."""
    def change(text: str) -> str:
        return _WORD.sub(lambda m: typo(m.group(), rng) if rng.random() < rate else m.group(), text)
    return map_text(fragment, change)


def assemble(fragments: Iterable[Part]) -> tuple[str, list[dict]]:
    """Return the final text and its spans; neighbouring parts are simply joined."""
    text = ""
    spans: list[dict] = []
    for part in fragments:
        start = len(text)
        text += part.text
        if part.label:
            spans.append({"start": start, "end": len(text), "label": part.label,
                          "text": part.text, "value": part.value})
    return text, spans
