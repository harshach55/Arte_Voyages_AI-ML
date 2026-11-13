"""Compare the first tagger (v1, public data only) with the second (v2, public + synthetic data).

Three sets, all labelled as optimistic:
- gold set: "seen before, optimistic". The rules were tuned on it, and the synthetic wording was
  written by someone who had read it (no copying; every synthetic example passed the hash check).
- held-out set 1: "seen before, optimistic", for the same reasons. Held-out set 2 is not touched.
- synthetic test split (10% of the synthetic enquiries, never trained or tuned on): "synthetic,
  optimistic", because it comes from the same generator as the training part.

Gold and held-out 1 are scored field by field (exact match), for rules only, each tagger alone,
and rules plus each tagger. The synthetic test split is scored span by span, per label.
Nothing is tuned here.

Usage:
    python -m src.uc2_extractor.training.compare_taggers --write-docs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ..evaluate import LANGS, load_gold
from ..extract import extract_profile
from ..metrics import ALL_FIELDS, fmt
from ..tagger import FIELDS_FILE_V2, MODEL_DIR, default_tagger, default_tagger_v2, default_tagger_v2_selective, tagger_profile
from . import train_crf
from .compare_tagger import count, pooled, table

SETS = {"gold": (Path("data/uc2_inquiries/gold"), "Gold set (60 enquiries): seen before, optimistic"),
        "heldout": (Path("data/uc2_inquiries/heldout"), "Held-out set 1 (45 enquiries): seen before, optimistic")}
SYNTHETIC_TITLE = "Synthetic test split: synthetic, optimistic"
MODES = ("rules only", "v1 only", "v2 only", "rules + v1", "rules + v2", "rules + v2 selective")
HEADING = "## Second tagger (public + synthetic data) compared with the first"
SUMMARY_FILE = MODEL_DIR / "comparison_v2.json"


def training_summary() -> dict:
    """Examples per source and language in each tagger's training and development parts."""
    out = {}
    for name, sources in (("v1", train_crf.SOURCES), ("v2", train_crf.SOURCES + (train_crf.SYNTHETIC,))):
        train, dev = train_crf.load_split(sources)
        out[name] = {"train": {f"{s} {l}": n for (s, l), n in sorted(Counter((e["source"], e["language"]) for e in train).items())},
                     "dev": {f"{s} {l}": n for (s, l), n in sorted(Counter((e["source"], e["language"]) for e in dev).items())}}
    out["synthetic_test"] = dict(Counter(e["language"] for e in train_crf.synthetic_test()))
    return out


def score_set(path: Path, v1, v2, v2s) -> dict:
    enquiries, labels = load_gold(path)
    extractors = {"rules only": extract_profile, "v1 only": lambda t: tagger_profile(t, v1),
                  "v2 only": lambda t: tagger_profile(t, v2), "rules + v1": lambda t: extract_profile(t, tagger=v1),
                  "rules + v2": lambda t: extract_profile(t, tagger=v2),
                  "rules + v2 selective": lambda t: extract_profile(t, tagger=v2s)}
    result = {"fields": {}, "languages": {}, "errors": {}}
    for mode in MODES:
        counts, errors = count(enquiries, labels, extractors[mode])
        for f in ALL_FIELDS:
            result["fields"].setdefault(f, {})[mode] = fmt(pooled(counts, [f], LANGS).f1)
        for lang in LANGS:
            result["languages"].setdefault(lang, {})[mode] = fmt(pooled(counts, ALL_FIELDS, [lang]).f1)
        result["errors"][mode] = sum(errors.values())
    return result


def score_synthetic(v1, v2) -> dict:
    examples = train_crf.synthetic_test()
    reports = {"v1": train_crf.score(v1, examples), "v2": train_crf.score(v2, examples)}
    out = {}
    for lang in LANGS:
        labels = sorted(set(reports["v1"][lang]) | set(reports["v2"][lang]), key=lambda x: (x == "ALL", x))
        out[lang] = {label: {"v1": reports["v1"][lang].get(label, {}).get("f1", 0.0),
                             "v2": reports["v2"][lang].get(label, {}).get("f1", 0.0),
                             "gold_spans": reports["v2"][lang].get(label, reports["v1"][lang].get(label, {})).get("gold_spans", 0)}
                     for label in labels}
    return out


def worse_fields(result: dict) -> list[str]:
    out = []
    for f, row in result["fields"].items():
        for a, b in (("rules + v1", "rules + v2"), ("v1 only", "v2 only"), ("rules only", "rules + v2 selective")):
            if row[a] != "-" and row[b] != "-" and float(row[b]) < float(row[a]):
                out.append(f"{f} ({b} {row[b]} against {a} {row[a]})")
    return out


def build_report(summary: dict) -> str:
    t = summary["training"]
    out = [HEADING, "",
           "How to read this: every number below is optimistic. The gold set and held-out set 1 were seen before "
           "(the rules were tuned on them, and the synthetic wording was written by someone who had read them). "
           "The synthetic test split comes from the same generator as the training data. The only real number is "
           "still the first rules-only run on held-out set 1 (0.81 / 0.88 / 0.86). Held-out set 2 was not touched. "
           "Nothing was tuned on any of these sets: c1 and c2 were chosen on the development split.", "",
           "v1 = first tagger, public data only. v2 = second tagger, public data plus 900 synthetic enquiries "
           "(80% training, 10% development, 10% test). French and German synthetic text has not been checked by a "
           "native speaker.", "",
           "rules + v2 selective = the second tagger may only fill empty fields in the units where it beat the rules "
           "on the gold set (exact-match F1, strictly better, decided on the gold set only, never on held-out set 1). "
           f"Units chosen: {', '.join(summary['selection']['chosen']) or 'none'}. "
           + ("With no unit chosen, this mode gives the same values as rules only. " if not summary['selection']['chosen'] else "")
           + "Because the list was chosen on the gold set, its gold numbers are optimistic twice over.", "",
           "Training data (training part, examples):", ""]
    rows = []
    for key in sorted(set(t["v1"]["train"]) | set(t["v2"]["train"])):
        rows.append([key, t["v1"]["train"].get(key, 0), t["v2"]["train"].get(key, 0)])
    out += [table(["Source and language", "v1", "v2"], rows), ""]
    for key in ("gold", "heldout"):
        result = summary[key]
        out += [f"### {SETS[key][1]}", "", "All fields pooled, F1 per language, and field errors:", ""]
        rows = [[mode] + [result["languages"][lang][mode] for lang in LANGS] + [result["errors"][mode]] for mode in MODES]
        out += [table(["Mode", "en", "fr", "de", "Field errors"], rows), "", "F1 per field (all languages):", ""]
        rows = [[f] + [result["fields"][f][mode] for mode in MODES] for f in ALL_FIELDS]
        out += [table(["Field"] + list(MODES), rows), ""]
        worse = worse_fields(result)
        out += ["Fields where v2 is worse than v1: " + ("; ".join(worse) if worse else "none") + ".", ""]
    out += [f"### {SYNTHETIC_TITLE}", "", "Span F1 per label (exact spans, as filtered in the app):", ""]
    labels = sorted({label for lang in LANGS for label in summary["synthetic"][lang]}, key=lambda x: (x == "ALL", x))
    rows = []
    for label in labels:
        row = [label]
        for lang in LANGS:
            cell = summary["synthetic"][lang].get(label)
            row += [f"{cell['v1']:.2f}", f"{cell['v2']:.2f}"] if cell else ["-", "-"]
        rows.append(row)
    sizes = ", ".join(f"{lang} {t['synthetic_test'].get(lang, 0)}" for lang in LANGS)
    out += [table(["Label"] + [f"{lang} {v}" for lang in LANGS for v in ("v1", "v2")], rows), "",
            f"Synthetic test enquiries: {sizes}. This split is small, so single labels can move a lot.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-docs", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    v1, v2, v2s = default_tagger(), default_tagger_v2(), default_tagger_v2_selective()
    selection = json.loads(FIELDS_FILE_V2.read_text(encoding="utf-8"))
    summary = {"training": training_summary(), "synthetic": score_synthetic(v1, v2),
               "selection": {"chosen": selection["chosen"], "rule": selection["rule"],
                             "decided_on": selection["decided_on"]},
               "titles": {**{k: v[1] for k, v in SETS.items()}, "synthetic": SYNTHETIC_TITLE}}
    for key, (path, _) in SETS.items():
        summary[key] = score_set(path, v1, v2, v2s)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = build_report(summary)
    print(report)
    if args.write_docs:
        _write(Path("docs/uc2_results.md"), report)
    return 0


def _write(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find(HEADING)
    if start == -1:
        text = text.rstrip() + "\n\n" + section
    else:
        following = text.find("\n## ", start + len(HEADING))
        text = text[:start] + section + (text[following + 1:] if following != -1 else "")
    path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
