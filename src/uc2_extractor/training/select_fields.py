"""Choose, on the GOLD SET ONLY, the fields where the second tagger beats the rules.

The tagger fills fields in units: one label fills one or more evaluation fields (a DATE span fills
start, end and unclear dates; a BUDGET span fills all five budget fields). For each unit, the exact-
match F1 of "rules only" and "rules + second tagger" is compared on the gold set, pooled over the
three languages. A unit is chosen only if rules + tagger is strictly better. Held-out set 1 is never
read here.

The field-selective mode then lets the tagger fill only the chosen units, and only where the rules
left them empty.

Writes models/uc2_crf/crf_tagger_v2_fields.json.

Usage:
    python -m src.uc2_extractor.training.select_fields
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ..evaluate import LANGS, load_gold
from ..extract import extract_profile
from ..metrics import Counts
from ..tagger import FIELDS_FILE_V2, default_tagger_v2
from .compare_tagger import count

GOLD = Path("data/uc2_inquiries/gold")
# unit -> (evaluation fields, tagger labels that fill them)
UNITS = {
    "destination": (("destination",), ("DESTINATION",)),
    "dates": (("start_date", "end_date", "dates_ambiguous"), ("DATE",)),
    "duration": (("duration",), ("DURATION",)),
    "party_size": (("party_size",), ("PARTY",)),
    "adults": (("adults",), ("ADULTS",)),
    "children": (("children",), ("CHILDREN",)),
    "ages": (("ages",), ("AGE",)),
    "budget": (("budget_min", "budget_max", "currency", "budget_basis", "budget_band"), ("BUDGET",)),
    "accommodation": (("accommodation",), ("ACCOMMODATION",)),
    "flights": (("flights",), ("FLIGHT", "ORIGIN")),
    "interests": (("interests",), ("INTEREST",)),
    "dietary": (("dietary",), ("DIETARY",)),
    "occasion": (("occasion",), ("OCCASION",)),
    "special_requests": (("special_requests",), ("SPECIAL_REQUEST",)),
}


def pooled(counts: dict, fields) -> Counts:
    total = Counts()
    for f in fields:
        for lang in LANGS:
            total.add(counts[(f, lang)])
    return total


def select(rules_counts: dict, tagger_counts: dict) -> dict:
    """Per unit: both scores, and whether the tagger is strictly better."""
    out = {}
    for unit, (fields, labels) in UNITS.items():
        r, t = pooled(rules_counts, fields), pooled(tagger_counts, fields)
        better = r.f1 is not None and t.f1 is not None and t.f1 > r.f1
        out[unit] = {"fields": list(fields), "labels": list(labels), "chosen": better,
                     "rules": {"tp": r.tp, "fp": r.fp, "fn": r.fn, "f1": r.f1},
                     "rules_plus_tagger": {"tp": t.tp, "fp": t.fp, "fn": t.fn, "f1": t.f1}}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", type=Path, default=GOLD)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "heldout" in str(args.gold_dir).replace("\\", "/").split("/"):
        print("The field list is chosen on the gold set only, never on a held-out set.")
        return 2
    enquiries, labels = load_gold(args.gold_dir)
    v2 = default_tagger_v2()
    rules_counts, _ = count(enquiries, labels, extract_profile)
    tagger_counts, _ = count(enquiries, labels, lambda text: extract_profile(text, tagger=v2))
    units = select(rules_counts, tagger_counts)
    chosen = [u for u, row in units.items() if row["chosen"]]
    info = {"decided_on": f"gold set only ({len(enquiries)} enquiries), never held-out set 1",
            "rule": "a unit is chosen if rules + second tagger has a strictly higher exact-match F1 than rules "
                    "only on the gold set, pooled over en, fr and de",
            "decided": datetime.now().isoformat(timespec="seconds"), "chosen": chosen,
            "labels": sorted({label for u in chosen for label in UNITS[u][1]}), "units": units}
    FIELDS_FILE_V2.write_text(json.dumps(info, indent=2), encoding="utf-8")
    for unit, row in units.items():
        r, t = row["rules"], row["rules_plus_tagger"]
        fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"{unit:<17} rules F1 {fmt(r['f1'])} (tp {r['tp']} fp {r['fp']} fn {r['fn']})   "
              f"rules + tagger F1 {fmt(t['f1'])} (tp {t['tp']} fp {t['fp']} fn {t['fn']})   "
              f"{'CHOSEN' if row['chosen'] else ''}")
    print(f"Chosen on the gold set: {chosen or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
