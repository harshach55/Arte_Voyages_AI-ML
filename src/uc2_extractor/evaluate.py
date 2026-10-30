"""Evaluate the rule-based extractor on the gold set.

Usage (from the project root):
    python -m src.uc2_extractor.evaluate
    python -m src.uc2_extractor.evaluate --gold-dir data/uc2_inquiries/gold --show-all

Reads enquiries.csv and labels.csv from the gold folder, runs the extractor on
every enquiry, prints exact and lenient precision/recall/F1 per field and per
language, and writes the tables plus a full error list to the output folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .extract import extract_profile
from .metrics import ALL_FIELDS, Counts, compare_field, fmt, normalise, profile_to_values
from .schema import check_evidence

DEFAULT_GOLD = Path("data/uc2_inquiries/gold")
DEFAULT_OUT = Path("data/uc2_inquiries/eval_output")
LANGS = ("en", "fr", "de")


def load_gold(gold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    read = dict(encoding="utf-8-sig", dtype=str, keep_default_na=False)
    enquiries = pd.read_csv(gold_dir / "enquiries.csv", **read)
    labels = pd.read_csv(gold_dir / "labels.csv", **read)
    missing = set(enquiries["id"]) - set(labels["id"])
    if missing:
        raise SystemExit(f"labels missing for: {sorted(missing)}")
    return enquiries, labels


def _rate_rows(counts: dict, keys) -> list[dict]:
    rows = []
    for key in keys:
        ex, le = counts[key]
        rows.append(
            dict(
                field=key,
                tp=ex.tp, fp=ex.fp, fn=ex.fn,
                precision=fmt(ex.precision), recall=fmt(ex.recall), f1=fmt(ex.f1),
                lenient_f1=fmt(le.f1),
                extra_on_empty=ex.extra_on_empty,
            )
        )
    return rows


def evaluate(enquiries: pd.DataFrame, labels: pd.DataFrame):
    gold = {(r.id, r.field): r.gold_value for r in labels.itertuples()}

    by_field = {f: (Counts(), Counts()) for f in ALL_FIELDS}
    by_field_lang = {(f, lang): (Counts(), Counts()) for f in ALL_FIELDS for lang in LANGS}
    by_lang = {lang: (Counts(), Counts()) for lang in LANGS}
    errors = []
    evidence_problems = []

    for row in enquiries.itertuples():
        profile = extract_profile(row.text)
        evidence_problems += [f"{row.id}: {p}" for p in check_evidence(profile, row.text)]
        pred = profile_to_values(profile)
        lang = row.language
        for f in ALL_FIELDS:
            g = gold.get((row.id, f), "")
            exact = compare_field(g, pred[f], f, lenient=False)
            lenient = compare_field(g, pred[f], f, lenient=True)
            for bucket in (by_field[f], by_field_lang.get((f, lang)), by_lang.get(lang)):
                if bucket:
                    bucket[0].add(exact.counts)
                    bucket[1].add(lenient.counts)
            if exact.missed or exact.extra:
                if exact.missed and exact.extra:
                    kind = "wrong"
                elif exact.missed:
                    kind = "missed"
                else:
                    kind = "extra"
                errors.append(
                    dict(
                        id=row.id, language=lang, field=f, error=kind,
                        gold=g, predicted=pred[f],
                        lenient_ok="yes" if not (lenient.missed or lenient.extra) else "no",
                    )
                )

    field_table = pd.DataFrame(_rate_rows(by_field, ALL_FIELDS))

    lang_rows = []
    for f in ALL_FIELDS:
        row = {"field": f}
        for lang in LANGS:
            ex, le = by_field_lang[(f, lang)]
            row[f"{lang}_f1"] = fmt(ex.f1)
            row[f"{lang}_lenient"] = fmt(le.f1)
        lang_rows.append(row)
    field_lang_table = pd.DataFrame(lang_rows)

    lang_table = pd.DataFrame(_rate_rows(by_lang, LANGS)).rename(columns={"field": "language"})
    lang_table.insert(1, "enquiries", [int((enquiries["language"] == l).sum()) for l in LANGS])

    error_table = pd.DataFrame(
        errors, columns=["id", "language", "field", "error", "gold", "predicted", "lenient_ok"]
    )
    return field_table, field_lang_table, lang_table, error_table, evidence_problems


def annotator_agreement(enquiries: pd.DataFrame, labels: pd.DataFrame) -> str:
    """Exact agreement between annotator 1 and annotator 2 where annotator 2 has worked."""
    sample = set(enquiries.loc[enquiries["second_annotator"].str.lower() == "yes", "id"])
    rows = labels[labels["id"].isin(sample)]
    done_ids = set(rows.loc[(rows["annotator_2_value"] != "") | (rows["annotator_2_notes"] != ""), "id"])
    if not done_ids:
        return f"Second annotator: not done yet (0 of {len(sample)} sampled enquiries)."
    rows = rows[rows["id"].isin(done_ids)]
    agree = sum(
        normalise(r.gold_value, r.field) == normalise(r.annotator_2_value, r.field)
        for r in rows.itertuples()
    )
    return (
        f"Second annotator: {len(done_ids)} of {len(sample)} sampled enquiries done; "
        f"exact agreement on {agree} of {len(rows)} field values ({agree / len(rows):.0%})."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show-all", action="store_true", help="print every error, not just the first 40")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    enquiries, labels = load_gold(args.gold_dir)
    field_table, field_lang_table, lang_table, error_table, evidence_problems = evaluate(enquiries, labels)

    print(f"Gold set: {len(enquiries)} enquiries "
          + ", ".join(f"{l}={int((enquiries['language'] == l).sum())}" for l in LANGS))
    print("\nOverall by language (all fields pooled, exact match)")
    print(lang_table.to_string(index=False))
    print("\nBy field (all languages, exact match; lenient F1 for comparison)")
    print(field_table.to_string(index=False))
    print("\nF1 by field and language (exact / lenient)")
    print(field_lang_table.to_string(index=False))
    print(f"\nEvidence spans that do not match the text: {len(evidence_problems)}")
    for p in evidence_problems:
        print("  " + p)
    print(annotator_agreement(enquiries, labels))

    print(f"\nErrors: {len(error_table)} field errors "
          f"({(error_table['error'] == 'missed').sum()} missed, "
          f"{(error_table['error'] == 'wrong').sum()} wrong, "
          f"{(error_table['error'] == 'extra').sum()} extra)")
    shown = error_table if args.show_all else error_table.head(40)
    print(shown.to_string(index=False))
    if not args.show_all and len(error_table) > 40:
        print(f"... {len(error_table) - 40} more (use --show-all or see errors.csv)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write = dict(index=False, encoding="utf-8-sig")
    field_table.to_csv(args.out_dir / "metrics_by_field.csv", **write)
    field_lang_table.to_csv(args.out_dir / "metrics_by_field_language.csv", **write)
    lang_table.to_csv(args.out_dir / "metrics_by_language.csv", **write)
    error_table.to_csv(args.out_dir / "errors.csv", **write)
    print(f"\nTables written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
