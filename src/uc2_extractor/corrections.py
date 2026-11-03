"""Saving reviewer corrections and turning them into labelled data.

Each correction is one JSON line in data/uc2_inquiries/corrections/corrections.jsonl
(git-ignored) with the enquiry text, the extracted profile, the corrected
profile, the language and a UTC timestamp.

Convert saved corrections into the held-out CSV format (enquiries.csv and
labels.csv), marked "from corrections, needs review":
    python -m src.uc2_extractor.corrections --to-csv data/uc2_inquiries/corrections/as_labels
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .metrics import ALL_FIELDS, profile_to_values
from .schema import ClientProfile, Duration, PartialDate

ROOT = Path(__file__).resolve().parents[2]
CORRECTIONS_FILE = ROOT / "data" / "uc2_inquiries" / "corrections" / "corrections.jsonl"
REVIEW_STATUS = "from corrections, needs review"
LIST_VALUE_FIELDS = ("destination", "accommodation", "flights", "interests", "dietary", "occasion",
                     "special_requests")


class CorrectionError(ValueError):
    """The corrected values cannot form a valid client profile."""


def _split(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split("|") if x.strip()]


def _number(value: str, field: str) -> Optional[float]:
    value = str(value or "").strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise CorrectionError(f"{field}: '{value}' is not a number")


def _int(value: str, field: str) -> Optional[int]:
    number = _number(value, field)
    if number is None:
        return None
    if not float(number).is_integer():
        raise CorrectionError(f"{field}: '{value}' must be a whole number")
    return int(number)


def _date(value: str, field: str) -> Optional[PartialDate]:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return PartialDate.from_str(value)
    except (ValueError, IndexError, ValidationError):
        raise CorrectionError(f"{field}: '{value}' is not a date like 2027-03-04, --03-04 or --03")


def _duration(value: str) -> Optional[Duration]:
    value = str(value or "").strip()
    if not value:
        return None
    m = re.fullmatch(r"(\d+)\s+(nights|days|weeks)", value)
    if not m:
        raise CorrectionError(f"duration: '{value}' must look like '10 nights', '12 days' or '2 weeks'")
    return Duration(value=int(m.group(1)), unit=m.group(2))


def values_to_profile(values: dict[str, str], extracted: Optional[ClientProfile] = None) -> ClientProfile:
    """Build a validated profile from flat field values (the gold-set format).

    Ambiguous dates can only be kept or removed, not added, because their
    readings come from the extraction.
    """
    unknown = set(values) - set(ALL_FIELDS)
    if unknown:
        raise CorrectionError(f"unknown fields: {sorted(unknown)}")
    v = {f: str(values.get(f, "") or "") for f in ALL_FIELDS}

    kept_ambiguous = []
    wanted = _split(v["dates_ambiguous"])
    available = {a.text: a for a in (extracted.dates.ambiguous if extracted else [])}
    for text in wanted:
        if text not in available:
            raise CorrectionError(
                f"dates_ambiguous: '{text}' was not found by the extraction; enter the date as start or end instead"
            )
        kept_ambiguous.append(available[text])

    try:
        ages = [_int(a, "ages") for a in _split(v["ages"])]
        return ClientProfile(
            language=v["language"].strip() or None,
            destination=_split(v["destination"]),
            dates={
                "start": _date(v["start_date"], "start_date"),
                "end": _date(v["end_date"], "end_date"),
                "duration": _duration(v["duration"]),
                "ambiguous": kept_ambiguous,
            },
            travellers={
                "party_size": _int(v["party_size"], "party_size"),
                "adults": _int(v["adults"], "adults"),
                "children": _int(v["children"], "children"),
                "ages": ages,
            },
            budget={
                "amount_min": _number(v["budget_min"], "budget_min"),
                "amount_max": _number(v["budget_max"], "budget_max"),
                "currency": v["currency"].strip().upper() or None,
                "basis": v["budget_basis"].strip() or None,
                "band": v["budget_band"].strip() or None,
            },
            **{f: _split(v[f]) for f in LIST_VALUE_FIELDS if f != "destination"},
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_input=False)
        )
        raise CorrectionError(problems)


def save_correction(text: str, extracted: ClientProfile, corrected_values: dict[str, str],
                    path: Path = CORRECTIONS_FILE) -> int:
    """Validate and append one correction; return the number of saved corrections."""
    if not text.strip():
        raise CorrectionError("the enquiry text is empty")
    corrected = values_to_profile(corrected_values, extracted)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": corrected.language or extracted.language,
        "text": text,
        "extracted": extracted.model_dump(mode="json"),
        "corrected": corrected.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return count_corrections(path)


def count_corrections(path: Path = CORRECTIONS_FILE) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_corrections(path: Path = CORRECTIONS_FILE) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_heldout_csv(path: Path, out_dir: Path) -> int:
    """Write enquiries.csv and labels.csv in the held-out format; return the number of rows."""
    records = load_corrections(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "enquiries.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "group_id", "language", "text", "review_status", "review_note", "second_annotator",
                    "checked_by", "checked_on"])
        for i, r in enumerate(records, 1):
            w.writerow([f"C{i:04d}", "", r.get("language") or "", r["text"], REVIEW_STATUS,
                        f"saved {r['timestamp']}", "no", "", ""])
    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "field", "gold_value", "annotator_2_value", "annotator_2_notes"])
        for i, r in enumerate(records, 1):
            values = profile_to_values(ClientProfile.model_validate(r["corrected"]))
            for field in ALL_FIELDS:
                w.writerow([f"C{i:04d}", field, values[field], "", ""])
    return len(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Turn saved corrections into held-out style CSV files.")
    parser.add_argument("--corrections", type=Path, default=CORRECTIONS_FILE)
    parser.add_argument("--to-csv", type=Path, required=True, help="output folder for enquiries.csv and labels.csv")
    args = parser.parse_args(argv)
    n = to_heldout_csv(args.corrections, args.to_csv)
    print(f"{n} corrections written to {args.to_csv} (review_status: '{REVIEW_STATUS}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
