"""Score held-out set 1 three ways: rules only, tagger only, and rules plus tagger.

This only scores. Nothing is tuned on this set, and the tagger never saw it. The rules were
tuned on it, so the comparison favours the rules.

Tagger only is scored on the fields the tagger can cover in each language. In French and
German that means dates only.

Usage:
    python -m src.uc2_extractor.training.compare_tagger --write-docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from ..evaluate import LANGS, load_gold
from ..extract import extract_profile
from ..metrics import ALL_FIELDS, Counts, compare_field, fmt, profile_to_values
from ..tagger import default_tagger, tagger_profile

HELD_OUT = Path("data/uc2_inquiries/heldout")
RESULTS_FILE = Path("docs/uc2_results.md")
HEADING = "## Trained tagger compared with the rules (held-out set 1)"

FIELDS_OF_LABEL = {
    "DESTINATION": ("destination",),
    "ORIGIN": ("flights",),
    "FLIGHT": ("flights",),
    "ACCOMMODATION": ("accommodation",),
    "DATE": ("start_date", "end_date", "dates_ambiguous"),
    "DURATION": ("duration",),
    "PARTY": ("party_size",),
    "ADULTS": ("adults",),
    "CHILDREN": ("children",),
}
MODES = ("rules only", "tagger only", "rules + tagger")


def covered_fields(labels: Iterable[str]) -> list[str]:
    """Fields the tagger can fill given its labels, in the usual field order."""
    fields = {f for label in labels for f in FIELDS_OF_LABEL.get(label, ())}
    return [f for f in ALL_FIELDS if f in fields]


def count(enquiries: pd.DataFrame, labels: pd.DataFrame, extract: Callable):
    """Exact-match counts per (field, language), and the number of field errors per language."""
    gold = {(r.id, r.field): r.gold_value for r in labels.itertuples()}
    counts = {(f, lang): Counts() for f in ALL_FIELDS for lang in LANGS}
    errors = {lang: 0 for lang in LANGS}
    for row in enquiries.itertuples():
        predicted = profile_to_values(extract(row.text))
        for f in ALL_FIELDS:
            result = compare_field(gold.get((row.id, f), ""), predicted[f], f, lenient=False)
            counts[(f, row.language)].add(result.counts)
            if result.missed or result.extra:
                errors[row.language] += 1
    return counts, errors


def pooled(counts: dict, fields: Iterable[str], langs: Iterable[str]) -> Counts:
    total = Counts()
    for lang in langs:
        for f in fields:
            total.add(counts[(f, lang)])
    return total


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def build_report(enquiries: pd.DataFrame, results: dict, cover: dict) -> str:
    rules, rules_errors = results["rules only"]
    only, only_errors = results["tagger only"]
    both, both_errors = results["rules + tagger"]
    sizes = ", ".join(f"{lang} {int((enquiries['language'] == lang).sum())}" for lang in LANGS)

    def f1(counts, fields, lang) -> str:
        return fmt(pooled(counts, fields, [lang]).f1)

    out = [HEADING, "",
           "How to read this: the tagger never saw held-out set 1, but the rules were tuned on it, so the "
           "comparison favours the rules. These are not a new real number; the only real number for this set "
           "is still the first rules-only run (0.81 / 0.88 / 0.86). Nothing was tuned on this set for this table: "
           "c1 and c2 were chosen on the development split of the public data, and the 0.5 confidence floor was "
           "fixed in advance.", "",
           f"Set: {len(enquiries)} synthetic enquiries ({sizes}). Exact match, F1. \"Added\" means values the tagger "
           "put into fields the rules left empty.", "",
           "All 21 fields pooled:", ""]
    rows = []
    for name, counts, errors in (("Rules only", rules, rules_errors), ("Rules + tagger", both, both_errors)):
        rows.append([name] + [f1(counts, ALL_FIELDS, lang) for lang in LANGS] + [sum(errors.values())])
    out += [table(["Mode", "en", "fr", "de", "Field errors"], rows), "", "Only the fields the tagger can cover in each language:", ""]
    rows = []
    for name, counts in (("Rules only", rules), ("Tagger only", only), ("Rules + tagger", both)):
        rows.append([name] + [f1(counts, cover[lang], lang) for lang in LANGS])
    out += [table(["Mode", "en (10 fields)", "fr (dates only)", "de (dates only)"], rows), "",
            "Per field and language. R = rules only, T = tagger only, R+T = rules + tagger, "
            "n/c = the tagger does not cover this field in this language:", ""]

    def cell(counts, field, lang, covered=True) -> str:
        return fmt(counts[(field, lang)].f1) if covered else "n/c"

    rows = []
    for field in ALL_FIELDS:
        row = [field]
        for lang in LANGS:
            row += [cell(rules, field, lang), cell(only, field, lang, field in cover[lang]), cell(both, field, lang)]
        rows.append(row)
    header = ["Field"] + [f"{lang} {m}" for lang in LANGS for m in ("R", "T", "R+T")]
    out += [table(header, rows), ""]

    fp = {lang: pooled(both, ALL_FIELDS, [lang]).fp - pooled(rules, ALL_FIELDS, [lang]).fp for lang in LANGS}
    tp = {lang: pooled(both, ALL_FIELDS, [lang]).tp - pooled(rules, ALL_FIELDS, [lang]).tp for lang in LANGS}
    out += ["What the tagger changed compared with rules only:", "",
            table(["", "en", "fr", "de"], [["Correct values added"] + [tp[lang] for lang in LANGS],
                                           ["Wrong values added"] + [fp[lang] for lang in LANGS]]), "",
            "Limits: English has nine labels, French and German only dates. The tagger has no training data for "
            "ages, budget, interests, dietary needs, occasion or special requests. All test enquiries are synthetic.",
            ""]
    return "\n".join(out)


def write_section(path: Path, section: str) -> None:
    """Add the section to the results file, or replace it if it is already there."""
    text = path.read_text(encoding="utf-8")
    start = text.find(HEADING)
    if start == -1:
        text = text.rstrip() + "\n\n" + section
    else:
        following = text.find("\n## ", start + len(HEADING))
        text = text[:start] + section + (text[following + 1:] if following != -1 else "")
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", type=Path, default=HELD_OUT)
    parser.add_argument("--write-docs", action="store_true", help=f"put the tables into {RESULTS_FILE}")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    enquiries, labels = load_gold(args.gold_dir)
    tagger = default_tagger()
    extractors = {"rules only": extract_profile, "tagger only": lambda text: tagger_profile(text, tagger),
                  "rules + tagger": lambda text: extract_profile(text, tagger=tagger)}
    results = {name: count(enquiries, labels, extractors[name]) for name in MODES}
    cover = {lang: covered_fields(tagger.labels_for(lang)) for lang in LANGS}
    report = build_report(enquiries, results, cover)
    print(report)
    if args.write_docs:
        write_section(RESULTS_FILE, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
