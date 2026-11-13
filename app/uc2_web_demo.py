"""Small browser demo for the travel enquiry extractor (standard library only).

Run from the project root:
    python app/uc2_web_demo.py            (then open http://localhost:8000)
    python app/uc2_web_demo.py --port 8080

Features: one enquiry or several (one per line), three modes (rules only, rules
plus the trained tagger, rules plus the optional model layer), a rule/crf/model
tag with confidence on every value, the pipeline with a status for each
component, and the evaluation scores.

The server only listens on this computer (127.0.0.1). The page inserts all
results with textContent, never as HTML, so pasted text cannot run as code.
The gold-set sample is read from data/uc2_inquiries/gold/ at start-up and is
not stored in this file; if that file is missing the button is disabled. If
the model or tagger files are missing, the demo says so and that mode is off.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from uc2_info import INFO_CSS, info_script  # noqa: E402

from src.uc2_extractor import lexicon as lx  # noqa: E402
from src.uc2_extractor.corrections import (  # noqa: E402
    CORRECTIONS_FILE,
    CorrectionError,
    count_corrections,
    save_correction,
)
from src.uc2_extractor.extract import extract_profile  # noqa: E402
from src.uc2_extractor.metrics import ALL_FIELDS, profile_to_values  # noqa: E402
from src.uc2_extractor.schema import ClientProfile, PartialDate, load_band_config  # noqa: E402
from src.uc2_extractor.tagger import INFO_FILE, check_tagger  # noqa: E402

GOLD_FILE = ROOT / "data" / "uc2_inquiries" / "gold" / "enquiries.csv"
GOLD_SAMPLE_ID = "P02-en"  # honeymoon enquiry with a GBP budget and an ambiguous date
MAX_CHARS = 20_000
MAX_BATCH = 50

SAMPLES = [
    {"label": "English: anniversary",
     "text": "Anniversary trip for two, late February, we love gardens, no early mornings"},
    {"label": "French: Rajasthan",
     "text": "Nous sommes deux adultes et deux enfants, nous voulons visiter le Rajasthan pendant dix nuits"},
    {"label": "German: 3/4 dates",
     "text": "Wir reisen vom 3/4 bis 10/4, zwei Erwachsene, vegetarisch"},
    {"label": "English: tagger fills gaps", "mode": "tagger",
     "text": "We would like to fly to Zanzibar on May 9th in coach, for two people."},
]

# Copied from docs/uc2_results.md (F1, all fields pooled, exact match).
SCORES = {
    "real": {
        "title": "Real number: first held-out run, rules only",
        "note": "45 new enquiries written and frozen before scoring, scored once on 3 November 2026.",
        "rows": [["Rules only", "0.81", "0.88", "0.86"]],
    },
    "optimistic": {
        "title": "Tuned on this set, optimistic",
        "note": "Rules or the model were changed after seeing errors on these same enquiries, "
                "so these numbers are higher than on new text.",
        "rows": [
            ["Held-out set, rules after improvements", "0.87", "0.91", "0.93"],
            ["Held-out set, rules + model (threshold 0.88)", "0.87", "0.92", "0.94"],
            ["Held-out set, rules + trained tagger", "0.87", "0.91", "0.93"],
            ["Gold set (60 enquiries), rules", "0.96", "0.98", "0.98"],
        ],
        "tagger": "The tagger never saw this set, but the rules were tuned on it, so the comparison favours the rules. "
                  "The tagger added 1 correct and 2 wrong values (74 field errors became 75).",
    },
    "next": "The next real number will come from a second held-out set written by colleagues, scored once at the end.",
}

TAGGER_HELD_OUT = "On held-out set 1 it added 1 correct and 2 wrong values."

# What happens to an enquiry, in order.
PIPELINE = [
    {"name": "Language detection", "kind": "rules", "detail": "Stopwords and letters: English, French or German."},
    {"name": "Trained tagger", "kind": "trained", "detail": "A CRF marks spans, each with a probability."},
    {"name": "Rules and normalisers", "kind": "rules",
     "detail": "Rule values win. Tagger spans fill only what is empty. Dates, numbers and currency are normalised."},
    {"name": "Optional embedding model", "kind": "pretrained",
     "detail": "Off by default. Fills empty label fields in rules + model mode."},
    {"name": "Schema output", "kind": "built", "detail": "Validated JSON, with the source text of each value."},
    {"name": "Review screen", "kind": "built", "detail": "A person corrects values. Corrections are saved locally."},
]
KIND_NAMES = {"trained": "Trained", "rules": "Rules", "pretrained": "Pretrained", "built": "Built", "pending": "Pending"}
OTHER_COMPONENTS = [
    {"name": "Language detection", "kind": "rules", "detail": "Stopword counts and letter hints."},
    {"name": "Field extractors and word lists", "kind": "rules", "detail": "Hand-written patterns for all fields."},
    {"name": "Date, number and currency normalisers", "kind": "rules", "detail": "Hand-written; also read the tagger's spans."},
    {"name": "Budget bands", "kind": "rules", "detail": "Edges are placeholders, not real company bands."},
    {"name": "Embedding model (MiniLM, ONNX)", "kind": "pretrained",
     "detail": "Used as downloaded, not trained here. Off by default."},
    {"name": "Schema, API and review screen", "kind": "built", "detail": "Ordinary software, no learning."},
    {"name": "Transformer tagger (XLM-RoBERTa)", "kind": "pending", "detail": "Not started. Waiting on the Colab decision."},
    {"name": "Synthetic training data", "kind": "pending",
     "detail": "Generator unfinished, so the tagger has no examples of ages, budget, interests, dietary, occasion or requests."},
    {"name": "Held-out set 2", "kind": "pending", "detail": "Colleagues have not written it yet; never scored."},
    {"name": "Label checks and French and German native review", "kind": "pending", "detail": "Not done."},
]


def components(tagger_available: bool) -> list[dict]:
    """The status list. The tagger's numbers are read from its training report, not typed in here."""
    tagger = {"name": "Slot tagger (CRF)", "kind": "pending", "detail": "Model file not found."}
    if tagger_available:
        tagger.update(kind="trained", detail="Trained here on public data.")
        try:
            info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
            f1 = [info["dev_report"][lang]["ALL"]["f1"] for lang in ("en", "fr", "de")]
            tagger["detail"] = (f"Trained here on {info['train_examples']:,} public examples. English has 9 labels; "
                                f"French and German have dates only. Development F1: en {f1[0]:.2f}, fr {f1[1]:.2f}, "
                                f"de {f1[2]:.2f}. {TAGGER_HELD_OUT}")
        except (OSError, KeyError, ValueError):
            pass
    return [tagger] + OTHER_COMPONENTS


# Field list and allowed values for the review-and-correct form.
CORRECTION_FIELDS = [
    {"name": "language", "label": "Language", "kind": "select", "options": ["", "en", "fr", "de"]},
    {"name": "destination", "label": "Destination", "kind": "list",
     "options": sorted(lx.DESTINATIONS), "free": True},
    {"name": "start_date", "label": "Start date", "kind": "text", "hint": "2027-03-04, --03-04 or --03"},
    {"name": "end_date", "label": "End date", "kind": "text", "hint": "2027-03-18, --03-18 or --03"},
    {"name": "duration", "label": "Trip length", "kind": "text", "hint": "10 nights, 12 days or 2 weeks"},
    {"name": "dates_ambiguous", "label": "Unclear dates", "kind": "list", "options": [], "remove_only": True},
    {"name": "party_size", "label": "Party size", "kind": "text", "hint": "number"},
    {"name": "adults", "label": "Adults", "kind": "text", "hint": "number"},
    {"name": "children", "label": "Children", "kind": "text", "hint": "number"},
    {"name": "ages", "label": "Ages", "kind": "list", "options": [], "free": True, "hint": "one age per value"},
    {"name": "budget_min", "label": "Budget from", "kind": "text", "hint": "number"},
    {"name": "budget_max", "label": "Budget up to", "kind": "text", "hint": "number"},
    {"name": "currency", "label": "Currency", "kind": "text", "hint": "USD, EUR, GBP, INR, CHF"},
    {"name": "budget_basis", "label": "Budget basis", "kind": "select", "options": ["", "per_trip", "per_person"]},
    {"name": "budget_band", "label": "Budget band", "kind": "select",
     "options": [""] + [b.label for b in load_band_config().band]},
    {"name": "accommodation", "label": "Accommodation", "kind": "list", "options": sorted(lx.ACCOMMODATION)},
    {"name": "flights", "label": "Flights", "kind": "list", "free": True,
     "options": sorted(lx.FLIGHT_CLASSES) + [f"departure from {c}" for c in sorted(lx.FLIGHT_ORIGINS)]},
    {"name": "interests", "label": "Interests", "kind": "list", "options": sorted(lx.INTERESTS)},
    {"name": "dietary", "label": "Dietary", "kind": "list", "options": sorted(lx.DIETARY)},
    {"name": "occasion", "label": "Occasion", "kind": "list", "options": sorted(lx.OCCASIONS)},
    {"name": "special_requests", "label": "Special requests", "kind": "list", "options": sorted(lx.SPECIAL_REQUESTS)},
]
assert [f["name"] for f in CORRECTION_FIELDS] == list(ALL_FIELDS)

LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German"}
ROWS = [
    ("language", "Language"),
    ("destination", "Destination"),
    ("dates", "Dates"),
    ("travellers", "Travellers"),
    ("budget", "Budget"),
    ("accommodation", "Accommodation"),
    ("flights", "Flights"),
    ("interests", "Interests"),
    ("dietary", "Dietary"),
    ("occasion", "Occasion"),
    ("special_requests", "Special requests"),
]
LIST_KEYS = ("destination", "accommodation", "flights", "interests", "dietary", "occasion", "special_requests")


def load_gold_sample() -> dict:
    sample = {"label": f"Gold set: {GOLD_SAMPLE_ID} (GBP)", "text": None}
    try:
        with open(GOLD_FILE, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["id"] == GOLD_SAMPLE_ID:
                    sample["text"] = row["text"]
                    break
    except OSError:
        pass
    return sample


def check_model() -> tuple[bool, str]:
    """Load the model once at start-up; report whether it is usable."""
    try:
        from src.uc2_extractor.semantic import DEFAULT_THRESHOLD, default_embedder

        default_embedder()(["warm-up"])
        return True, f"Model available (threshold {DEFAULT_THRESHOLD})."
    except Exception as exc:  # missing files or packages
        return False, f"Model not available, so the demo runs rules only ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# Turning a profile into readable rows
# ---------------------------------------------------------------------------


def _date(d: PartialDate) -> str:
    parts = []
    if d.day is not None:
        parts.append(str(d.day))
    parts.append(calendar.month_name[d.month])
    if d.year is not None:
        parts.append(str(d.year))
    return " ".join(parts)


def _amount(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def describe_sections(p: ClientProfile) -> dict[str, str]:
    """Readable text for the language, dates, travellers and budget sections."""
    out = {"language": "", "dates": "", "travellers": "", "budget": ""}
    if p.language:
        out["language"] = f"{LANGUAGE_NAMES[p.language]} ({p.language})"

    d = p.dates
    bits = []
    if d.start and d.end:
        bits.append(f"{_date(d.start)} to {_date(d.end)}")
    elif d.start:
        bits.append(f"from {_date(d.start)}")
    elif d.end:
        bits.append(f"until {_date(d.end)}")
    if d.duration:
        bits.append(f"{d.duration.value} {d.duration.unit if d.duration.value != 1 else d.duration.unit[:-1]}")
    for a in d.ambiguous:
        readings = " or ".join(_date(r) for r in a.readings)
        bits.append(f"'{a.text}' is unclear ({readings})")
    out["dates"] = "; ".join(bits)

    t = p.travellers
    bits = []
    if t.party_size is not None:
        bits.append(f"{t.party_size} {'person' if t.party_size == 1 else 'people'}")
    split = []
    if t.adults is not None:
        split.append(f"{t.adults} adult{'s' if t.adults != 1 else ''}")
    if t.children is not None:
        split.append(f"{t.children} child{'ren' if t.children != 1 else ''}")
    if split:
        bits.append(", ".join(split) if not bits else f"({', '.join(split)})")
    if t.ages:
        bits.append("ages " + ", ".join(str(a) for a in t.ages))
    out["travellers"] = " ".join(bits)

    b = p.budget
    if b.amount_min is not None or b.amount_max is not None:
        cur = f" {b.currency}" if b.currency else " (currency not stated)"
        if b.amount_min is not None and b.amount_max is not None and b.amount_min != b.amount_max:
            text = f"{_amount(b.amount_min)} to {_amount(b.amount_max)}{cur}"
        elif b.amount_min is None:
            text = f"up to {_amount(b.amount_max)}{cur}"
        elif b.amount_max is None:
            text = f"at least {_amount(b.amount_min)}{cur}"
        else:
            text = f"{_amount(b.amount_min)}{cur}"
        if b.basis:
            text += ", " + b.basis.replace("_", " ")
        if b.band:
            text += f"; band {b.band} (placeholder)"
        out["budget"] = text
    return out


def section_source(p: ClientProfile, key: str) -> tuple[str, Optional[float]]:
    """Where a section's text came from. A section can mix rule values and tagger values."""
    crf = [x.confidence for x in p.provenance if x.field == key and x.source == "crf"]
    if not crf:
        return "rule", None
    has_rule = any(x.field == key and x.source == "rule" for x in p.provenance)
    return ("rule+crf" if has_rule else "crf"), min(crf)


def build_rows(p: ClientProfile) -> list[dict]:
    """One row per field: items with their source and confidence, and the evidence."""
    sections = describe_sections(p)
    source = {(x.field, x.value): x for x in p.provenance}
    evidence: dict[str, list[dict]] = {}
    for ev in p.evidence:
        evidence.setdefault(ev.field, []).append({"text": ev.text, "source": ev.source})

    rows = []
    for key, label in ROWS:
        if key in LIST_KEYS:
            items = []
            for value in getattr(p, key):
                prov = source.get((key, value))
                items.append({
                    "text": value,
                    "source": prov.source if prov else "rule",
                    "confidence": prov.confidence if prov else None,
                })
        else:
            origin, confidence = section_source(p, key)
            items = [{"text": sections[key], "source": origin, "confidence": confidence}] if sections[key] else []
        rows.append({"label": label, "items": items, "evidence": evidence.get(key, [])})
    return rows


def build_response(text: str, use_model: bool, use_tagger: bool = False) -> dict:
    profile = extract_profile(text, model=use_model, tagger=use_tagger)
    flags = [{"field": f.field.replace("_", " "), "kind": f.kind, "message": f.message} for f in profile.flags]
    return {"rows": build_rows(profile), "flags": flags, "profile": profile.model_dump(mode="json"),
            "values": profile_to_values(profile)}


def handle_save(data: Optional[dict], model_available: bool, path: Path = CORRECTIONS_FILE,
                tagger_available: bool = False) -> tuple[int, dict]:
    """Save one correction. The extracted profile is recomputed from the text on
    the server, so the saved 'extracted' part cannot be altered in the browser."""
    if not isinstance(data, dict):
        return 400, {"error": "request must be JSON with 'text', 'mode' and 'corrected'"}
    text = str(data.get("text", "")).strip()
    corrected = data.get("corrected")
    if not text or len(text) > MAX_CHARS:
        return 400, {"error": "enquiry is empty or too long"}
    if not isinstance(corrected, dict) or not all(isinstance(v, str) for v in corrected.values()):
        return 400, {"error": "'corrected' must map field names to text values"}
    use_model = data.get("mode") == "model" and model_available
    use_tagger = data.get("mode") == "tagger" and tagger_available
    extracted = extract_profile(text, model=use_model, tagger=use_tagger)
    try:
        count = save_correction(text, extracted, corrected, path)
    except CorrectionError as exc:
        return 400, {"error": f"Not saved: {exc}"}
    return 200, {"count": count}


def build_batch(texts: list[str], use_model: bool, use_tagger: bool = False) -> list[dict]:
    """A short summary row per enquiry for the several-enquiries table."""
    out = []
    for i, text in enumerate(texts, 1):
        p = extract_profile(text, model=use_model, tagger=use_tagger)
        sections = describe_sections(p)
        source = {(x.field, x.value): x for x in p.provenance}

        def shown(key):  # batch cells are plain text, so a tagger value is marked in the text
            return f"{sections[key]} (crf)" if sections[key] and section_source(p, key)[0] != "rule" else sections[key]

        def items(key):
            return [{"text": v, "source": source[(key, v)].source if (key, v) in source else "rule",
                     "confidence": source[(key, v)].confidence if (key, v) in source else None}
                    for v in getattr(p, key)]

        details = []
        for key in ("accommodation", "flights", "interests", "dietary", "occasion", "special_requests"):
            for item in items(key):
                details.append({**item, "text": f"{key.replace('_', ' ')}: {item['text']}"})
        out.append({
            "n": i,
            "text": text,
            "language": p.language or "",
            "destination": items("destination"),
            "dates": shown("dates"),
            "travellers": shown("travellers"),
            "budget": shown("budget"),
            "details": details,
            "flags": len(p.flags),
        })
    return out


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Travel Enquiry Extractor</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; font-size: 20px; line-height: 1.45;
         color: #1a1a1a; background: #ffffff; margin: 0; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px 20px 40px; }
  h1 { font-size: 32px; margin: 0 0 12px; }
  h2 { font-size: 24px; margin: 28px 0 8px; }
  textarea { width: 100%; box-sizing: border-box; height: 170px; font-size: 20px; padding: 12px;
             border: 2px solid #888; border-radius: 6px; font-family: inherit; }
  .controls { margin: 12px 0 8px; display: flex; flex-wrap: wrap; gap: 18px; align-items: center; }
  .controls label { cursor: pointer; }
  .buttons { margin: 10px 0 16px; display: flex; flex-wrap: wrap; gap: 10px; }
  button { font-size: 18px; padding: 10px 16px; border-radius: 6px; border: 2px solid #555;
           background: #f2f2f2; cursor: pointer; }
  button.primary { background: #1f4e8c; color: #fff; border-color: #1f4e8c; font-weight: bold; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid #ccc; }
  th { background: #eef1f5; }
  td.field { font-weight: bold; width: 16%; white-space: nowrap; }
  td.source { color: #444; font-style: italic; }
  .empty { color: #9a9a9a; font-style: normal; }
  .item { display: inline-block; margin: 0 10px 4px 0; }
  .tag { font-size: 14px; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-style: normal;
         vertical-align: middle; white-space: nowrap; }
  .tag.rule { background: #e6e6e6; color: #333; }
  .tag.model { background: #dbe8ff; color: #123a7a; border: 1px solid #7a9fd6; }
  .tag.crf { background: #d7f0dc; color: #14532d; border: 1px solid #5fa873; }
  .stages { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; padding: 0; margin: 10px 0; align-items: stretch; }
  .stage { flex: 1 1 150px; border: 2px solid #1f4e8c; border-radius: 6px; padding: 8px 10px; background: #f5f8fc;
           font-size: 16px; line-height: 1.3; }
  .stage strong { display: block; font-size: 17px; margin-bottom: 2px; }
  .arrow { align-self: center; font-size: 22px; color: #1f4e8c; }
  .badge { font-size: 14px; padding: 1px 7px; border-radius: 4px; white-space: nowrap; display: inline-block; }
  .badge.trained { background: #d7f0dc; color: #14532d; }
  .badge.rules { background: #e6e6e6; color: #333; }
  .badge.pretrained { background: #dbe8ff; color: #123a7a; }
  .badge.built { background: #ece6f7; color: #3f2a78; }
  .badge.pending { background: #fff4d6; color: #7a5200; }
  .notice { padding: 10px 14px; border-radius: 6px; background: #f2f2f2; border: 1px solid #ccc; }
  .notice.warn { background: #fff4d6; border-color: #e0a100; }
  .flags { margin-top: 22px; padding: 14px 18px; background: #fff4d6; border: 2px solid #e0a100;
           border-radius: 6px; }
  .flags h3 { font-size: 22px; margin: 0 0 8px; color: #7a5200; }
  .flags ul { margin: 0; padding-left: 24px; }
  .kind { font-weight: bold; color: #7a5200; }
  pre { background: #f6f6f6; border: 1px solid #ccc; padding: 12px; font-size: 15px;
        overflow-x: auto; white-space: pre-wrap; }
  .error { color: #a00000; font-weight: bold; }
  .scores table { max-width: 900px; }
  .scores .real th { background: #dff0d8; }
  .scores .opt th { background: #f7e7c6; }
  .small { font-size: 17px; color: #555; }
  .review { margin-top: 26px; padding: 16px 18px; border: 2px solid #1f4e8c; border-radius: 6px; }
  .review h3 { margin: 0 0 6px; font-size: 22px; }
  .review table td { vertical-align: middle; }
  .review input[type=text], .review select { font-size: 18px; padding: 6px 8px; border: 1px solid #888;
             border-radius: 4px; min-width: 260px; }
  .chip { display: inline-block; margin: 2px 6px 2px 0; padding: 3px 4px 3px 10px; background: #eef1f5;
          border: 1px solid #9aa7b8; border-radius: 14px; }
  .chip button { font-size: 15px; padding: 0 7px; margin-left: 4px; border-radius: 10px; border-width: 1px; }
  .addbox { display: inline-flex; gap: 6px; margin-top: 4px; }
  .addbox button { font-size: 16px; padding: 4px 10px; }
  .hint { font-size: 15px; color: #666; }
  .saved { font-weight: bold; color: #1d6b2c; }
  footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid #ccc; color: #555; font-size: 17px; }
__INFO_CSS__
</style>
</head>
<body>
<main>
  <h1>Travel Enquiry Extractor (English, French, German)</h1>
  <p id="model-status" class="notice"></p>

  <div class="controls">
    <span><strong>Input:</strong>
      <label><input type="radio" name="input" value="one" checked> one enquiry</label>
      <label><input type="radio" name="input" value="many"> several enquiries, one per line</label><span data-info-slot="several"></span>
    </span>
    <span><strong>Mode:</strong><span data-info-slot="mode"></span>
      <label><input type="radio" name="mode" value="rules" checked> rules only</label>
      <label id="tagger-choice"><input type="radio" name="mode" value="tagger"> rules + trained tagger</label>
      <label id="model-choice"><input type="radio" name="mode" value="model"> rules + model</label>
    </span>
  </div>
  <textarea id="enquiry" maxlength="20000" aria-label="Enquiry text"></textarea>
  <div class="buttons">
    <button class="primary" id="extract">Extract</button>
    <span id="samples"></span>
  </div>
  <p id="status"></p>

  <div id="single" hidden>
    <p class="small">Tags: <span class="tag rule">rule</span><span data-info-slot="tag_rule"></span>
      <span class="tag model">model 0.xx</span><span data-info-slot="tag_model"></span>
      <span class="tag crf">crf 0.xx</span><span data-info-slot="tag_crf"></span></p>
    <table>
      <thead><tr><th>Field</th><th>Value</th><th>Taken from the text<span data-info-slot="source"></span></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="flags">
      <h3>Check these<span data-info-slot="flags"></span></h3>
      <ul id="flags"></ul>
    </div>
    <p><button id="toggle-json">Show raw JSON</button></p>
    <pre id="json" hidden></pre>

    <div class="review">
      <h3>Review and correct</h3>
      <p class="small">Edit a value, remove a wrong one with &#x2715;, or add a missing one, then save.
        Lists can take several values.</p>
      <table><tbody id="review-rows"></tbody></table>
      <p>
        <button class="primary" id="save-corrections">Save corrections</button><span data-info-slot="save"></span>
        <button id="reset-corrections">Undo my changes</button>
        <span id="save-status"></span>
      </p>
    </div>
  </div>
  <p class="small">Saved corrections: <strong id="correction-count">0</strong><span data-info-slot="save"></span></p>

  <div id="batch" hidden>
    <table>
      <thead><tr><th>#</th><th>Lang.</th><th>Destination</th><th>Dates</th><th>Travellers</th>
        <th>Budget</th><th>Other details</th><th>Flags</th></tr></thead>
      <tbody id="batch-rows"></tbody>
    </table>
  </div>

  <section>
    <h2>Pipeline<span data-info-slot="pipeline"></span></h2>
    <ol id="pipeline" class="stages"></ol>
    <p class="small">The rules run on every enquiry and their values always win. The tagger only fills what the rules
      left empty. It knows nine labels in English and only dates in French and German.</p>
    <h3>Components</h3>
    <table>
      <thead><tr><th>Component</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody id="components"></tbody>
    </table>
  </section>

  <section class="scores">
    <h2>How well it works</h2>
    <p class="small">F1 score (1.00 would be perfect), all fields pooled, exact match.</p>
    <div id="scores"></div>
  </section>

  <footer>Rules, a trained tagger and an optional model layer. Scored on our own synthetic sets. Budget bands are placeholders.</footer>
</main>
<script>
__INFO_JS__
const SAMPLES = __SAMPLES__;
const SCORES = __SCORES__;
const MODEL = __MODEL__;
const TAGGER = __TAGGER__;
const FIELDS = __FIELDS__;
const PIPELINE = __PIPELINE__;
const COMPONENTS = __COMPONENTS__;
const KIND_NAMES = __KIND_NAMES__;
let lastRequest = null;      // {text, mode} of the enquiry shown in the review form
let originalValues = null;   // values as extracted
let reviewValues = null;     // values being edited
const box = document.getElementById("enquiry");
const statusLine = document.getElementById("status");

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function choice(name) { return document.querySelector('input[name="' + name + '"]:checked').value; }
function tag(item) {
  if (item.source === "model") return el("span", "model " + item.confidence.toFixed(2), "tag model");
  if (item.source === "crf") return el("span", "crf " + item.confidence.toFixed(2), "tag crf");
  if (item.source === "rule+crf") return el("span", "rule + crf " + item.confidence.toFixed(2), "tag crf");
  return el("span", "rule", "tag rule");
}
function itemList(items) {
  const td = el("td");
  if (!items.length) { td.appendChild(el("span", "\\u2014", "empty")); return td; }
  for (const it of items) {
    const span = el("span", null, "item");
    span.appendChild(document.createTextNode(it.text));
    span.appendChild(tag(it));
    td.appendChild(span);
  }
  return td;
}

// model status and toggle
const status = document.getElementById("model-status");
status.textContent = MODEL.message + " " + TAGGER.message;
if (!MODEL.available) {
  status.className = "notice warn";
  const radio = document.querySelector('#model-choice input');
  radio.disabled = true;
  document.getElementById("model-choice").title = "Model files not found: rules only";
}
if (!TAGGER.available) {
  status.className = "notice warn";
  document.querySelector('#tagger-choice input').disabled = true;
  document.getElementById("tagger-choice").title = "Tagger file not found: rules only";
}

// pipeline and component status
function badge(kind) { return el("span", KIND_NAMES[kind], "badge " + kind); }
const stages = document.getElementById("pipeline");
PIPELINE.forEach((s, i) => {
  if (i) stages.appendChild(el("li", "→", "arrow"));
  const li = el("li", null, "stage");
  li.appendChild(el("strong", (i + 1) + ". " + s.name));
  li.appendChild(badge(s.kind));
  li.appendChild(el("div", s.detail));
  stages.appendChild(li);
});
const componentRows = document.getElementById("components");
for (const c of COMPONENTS) {
  const tr = el("tr");
  tr.appendChild(el("td", c.name, "field"));
  const td = el("td");
  td.appendChild(badge(c.kind));
  tr.appendChild(td);
  tr.appendChild(el("td", c.detail));
  componentRows.appendChild(tr);
}

// sample buttons
const sampleSpan = document.getElementById("samples");
for (const s of SAMPLES) {
  const b = el("button", s.label);
  if (s.text === null) {
    b.disabled = true;
    b.title = "Gold set file not found in data/uc2_inquiries/gold/";
  } else {
    b.addEventListener("click", () => {
      document.querySelector('input[name="input"][value="one"]').checked = true;
      const wanted = s.mode ? document.querySelector('input[name="mode"][value="' + s.mode + '"]') : null;
      if (wanted && !wanted.disabled) wanted.checked = true;
      box.value = s.text; extract();
    });
  }
  sampleSpan.appendChild(b);
  sampleSpan.appendChild(document.createTextNode(" "));
}

// scores
const scoreBox = document.getElementById("scores");
for (const [key, cls] of [["real", "real"], ["optimistic", "opt"]]) {
  const part = SCORES[key];
  const heading = el("h3", part.title);
  heading.appendChild(infoButton(key === "real" ? "real_number" : "optimistic"));
  scoreBox.appendChild(heading);
  scoreBox.appendChild(el("p", part.note, "small"));
  if (part.tagger) scoreBox.appendChild(el("p", part.tagger, "small"));
  const table = el("table", null, cls);
  const head = el("tr");
  for (const h of ["", "English", "French", "German"]) head.appendChild(el("th", h));
  table.appendChild(head);
  for (const row of part.rows) {
    const tr = el("tr");
    row.forEach((v, i) => tr.appendChild(el(i === 0 ? "td" : "td", v, i === 0 ? "field" : "")));
    table.appendChild(tr);
  }
  scoreBox.appendChild(table);
}
scoreBox.appendChild(el("p", SCORES.next, "small"));

function showSingle(data) {
  const tbody = document.getElementById("rows");
  tbody.replaceChildren();
  for (const r of data.rows) {
    const tr = el("tr");
    const name = el("td", r.label, "field");
    for (const key of infoKeys(r.label, r.items.map(i => i.text).join(" "))) name.appendChild(infoButton(key));
    tr.appendChild(name);
    tr.appendChild(itemList(r.items));
    const src = el("td", null, r.evidence.length ? "source" : "source empty");
    if (!r.evidence.length) src.textContent = "\\u2014";
    r.evidence.forEach((ev, i) => {
      if (i) src.appendChild(document.createTextNode("  \\u00b7  "));
      src.appendChild(document.createTextNode("\\u201c" + ev.text + "\\u201d"));
      if (ev.source === "model") src.appendChild(el("span", "model", "tag model"));
      if (ev.source === "crf") src.appendChild(el("span", "crf", "tag crf"));
    });
    tr.appendChild(src);
    tbody.appendChild(tr);
  }
  const list = document.getElementById("flags");
  list.replaceChildren();
  if (data.flags.length === 0) list.appendChild(el("li", "Nothing flagged."));
  for (const f of data.flags) {
    const li = el("li");
    li.appendChild(el("span", f.kind + ": ", "kind"));
    li.appendChild(document.createTextNode(f.field + " \\u2013 " + f.message));
    list.appendChild(li);
  }
  document.getElementById("json").textContent = JSON.stringify(data.profile, null, 2);
  originalValues = data.values;
  reviewValues = Object.assign({}, data.values);
  document.getElementById("save-status").textContent = "";
  renderReview();
  document.getElementById("batch").hidden = true;
  document.getElementById("single").hidden = false;
}

// ----- review and correct -----
function splitList(v) { return (v || "").split("|").map(x => x.trim()).filter(x => x); }

function renderReview() {
  const tbody = document.getElementById("review-rows");
  tbody.replaceChildren();
  for (const f of FIELDS) {
    const tr = el("tr");
    const name = el("td", f.label, "field");
    const infoKey = {language: "language", dates_ambiguous: "unclear_dates", budget_band: "budget_band"}[f.name];
    if (infoKey) name.appendChild(infoButton(infoKey));
    tr.appendChild(name);
    const td = el("td");
    if (f.kind === "select") {
      const sel = el("select");
      const current = reviewValues[f.name] || "";
      const options = f.options.includes(current) ? f.options : f.options.concat([current]);
      for (const o of options) {
        const opt = el("option", o === "" ? "\\u2014 (empty)" : o);
        opt.value = o;
        if (o === current) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.addEventListener("change", () => { reviewValues[f.name] = sel.value; });
      sel.setAttribute("aria-label", f.label);
      td.appendChild(sel);
    } else if (f.kind === "text") {
      const input = el("input");
      input.type = "text";
      input.value = reviewValues[f.name] || "";
      input.placeholder = f.hint || "";
      input.setAttribute("aria-label", f.label);
      input.addEventListener("input", () => { reviewValues[f.name] = input.value.trim(); });
      td.appendChild(input);
    } else {
      const items = splitList(reviewValues[f.name]);
      if (!items.length) td.appendChild(el("span", "\\u2014 ", "empty"));
      items.forEach((item, i) => {
        const chip = el("span", null, "chip");
        chip.appendChild(document.createTextNode(item));
        const x = el("button", "\\u2715");
        x.title = "Remove " + item;
        x.setAttribute("aria-label", "Remove " + item);
        x.addEventListener("click", () => {
          items.splice(i, 1);
          reviewValues[f.name] = items.join(" | ");
          renderReview();
        });
        chip.appendChild(x);
        td.appendChild(chip);
      });
      if (!f.remove_only) {
        const wrap = el("span", null, "addbox");
        const input = el("input");
        input.type = "text";
        input.placeholder = f.hint || (f.free ? "type or pick a value" : "pick a value");
        input.setAttribute("aria-label", "Add " + f.label);
        if (f.options.length) {
          const list = el("datalist");
          list.id = "opts-" + f.name;
          for (const o of f.options) { const opt = el("option"); opt.value = o; list.appendChild(opt); }
          wrap.appendChild(list);
          input.setAttribute("list", list.id);
        }
        const add = el("button", "Add");
        const doAdd = () => {
          const v = input.value.trim().replace(/\\|/g, " ");
          if (!v) return;
          if (!f.free && f.options.length && !f.options.includes(v)) {
            document.getElementById("save-status").textContent = "'" + v + "' is not one of the " + f.label.toLowerCase() + " labels.";
            return;
          }
          if (!items.includes(v)) items.push(v);
          reviewValues[f.name] = items.join(" | ");
          renderReview();
        };
        add.addEventListener("click", doAdd);
        input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); doAdd(); } });
        wrap.appendChild(input);
        wrap.appendChild(add);
        td.appendChild(wrap);
      }
      if (f.remove_only) td.appendChild(el("span", " (can only be removed)", "hint"));
    }
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

async function refreshCount() {
  try {
    const res = await fetch("/corrections_count");
    document.getElementById("correction-count").textContent = String((await res.json()).count);
  } catch (err) { /* the counter is only informative */ }
}

async function saveCorrections() {
  const out = document.getElementById("save-status");
  if (!lastRequest || !reviewValues) { out.textContent = "Extract an enquiry first."; return; }
  out.className = "";
  out.textContent = "Saving\\u2026";
  try {
    const data = await post("/save_correction", {text: lastRequest.text, mode: lastRequest.mode, corrected: reviewValues});
    document.getElementById("correction-count").textContent = String(data.count);
    out.textContent = "Saved.";
    out.className = "saved";
  } catch (err) {
    out.textContent = err.message;
    out.className = "error";
  }
}

document.getElementById("save-corrections").addEventListener("click", saveCorrections);
document.getElementById("reset-corrections").addEventListener("click", () => {
  if (!originalValues) return;
  reviewValues = Object.assign({}, originalValues);
  document.getElementById("save-status").textContent = "";
  renderReview();
});
refreshCount();

function cell(text) { return text ? el("td", text) : el("td", "\\u2014", "empty"); }

function showBatch(rows) {
  const tbody = document.getElementById("batch-rows");
  tbody.replaceChildren();
  for (const r of rows) {
    const tr = el("tr");
    const n = el("td", String(r.n), "field");
    n.title = r.text;
    tr.appendChild(n);
    tr.appendChild(cell(r.language));
    tr.appendChild(itemList(r.destination));
    tr.appendChild(cell(r.dates));
    tr.appendChild(cell(r.travellers));
    tr.appendChild(cell(r.budget));
    tr.appendChild(itemList(r.details));
    tr.appendChild(el("td", String(r.flags)));
    tbody.appendChild(tr);
  }
  document.getElementById("single").hidden = true;
  document.getElementById("batch").hidden = false;
}

async function post(url, body) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                                body: JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function extract() {
  const text = box.value.trim();
  if (!text) { statusLine.textContent = "Please paste an enquiry first."; return; }
  const useModel = choice("mode") === "model" && MODEL.available;
  const useTagger = choice("mode") === "tagger" && TAGGER.available;
  statusLine.textContent = "Extracting\\u2026";
  statusLine.className = "";
  try {
    if (choice("input") === "many") {
      const texts = text.split(/\\r?\\n/).map(t => t.trim()).filter(t => t);
      showBatch((await post("/extract_batch", {texts, model: useModel, tagger: useTagger})).rows);
    } else {
      const data = await post("/extract", {text, model: useModel, tagger: useTagger});
      lastRequest = {text, mode: useTagger ? "tagger" : useModel ? "model" : "rules"};
      showSingle(data);
    }
    statusLine.textContent = "Mode used: " + (useTagger ? "rules + trained tagger" : useModel ? "rules + model" : "rules only");
  } catch (err) {
    statusLine.textContent = "Error: " + err.message;
    statusLine.className = "error";
  }
}

document.getElementById("extract").addEventListener("click", extract);
document.getElementById("toggle-json").addEventListener("click", (e) => {
  const pre = document.getElementById("json");
  pre.hidden = !pre.hidden;
  e.target.textContent = pre.hidden ? "Show raw JSON" : "Hide raw JSON";
});
</script>
</body>
</html>
"""


def _script_json(data) -> str:
    # json.dumps output is safe inside a script block once "</" is escaped
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_page(samples: list[dict], model_available: bool, model_message: str,
                tagger_available: bool = False, tagger_message: str = "Trained tagger not loaded.") -> bytes:
    page = PAGE.replace("__SAMPLES__", _script_json(samples))
    page = page.replace("__SCORES__", _script_json(SCORES))
    page = page.replace("__MODEL__", _script_json({"available": model_available, "message": model_message}))
    page = page.replace("__TAGGER__", _script_json({"available": tagger_available, "message": tagger_message}))
    page = page.replace("__FIELDS__", _script_json(CORRECTION_FIELDS))
    page = page.replace("__INFO_CSS__", INFO_CSS).replace("__INFO_JS__", info_script(_script_json))
    page = page.replace("__PIPELINE__", _script_json(PIPELINE))
    page = page.replace("__COMPONENTS__", _script_json(components(tagger_available)))
    page = page.replace("__KIND_NAMES__", _script_json(KIND_NAMES))
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    page: bytes = b""
    model_available: bool = False
    tagger_available: bool = False
    corrections_path: Path = CORRECTIONS_FILE

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self, limit: int):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > limit:
            return None
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, self.page, "text/html; charset=utf-8")
        elif self.path == "/corrections_count":
            self._json(200, {"count": count_corrections(self.corrections_path)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/extract":
            data = self._read_json(MAX_CHARS * 4)
            text = str(data.get("text", "")).strip() if data else ""
            if not text or len(text) > MAX_CHARS:
                self._json(400, {"error": "enquiry is empty or too long"})
                return
            use_model = bool(data.get("model")) and self.model_available
            use_tagger = bool(data.get("tagger")) and self.tagger_available
            self._json(200, build_response(text, use_model, use_tagger))
        elif self.path == "/extract_batch":
            data = self._read_json(MAX_CHARS * 4 * 2)
            texts = data.get("texts") if data else None
            if not isinstance(texts, list):
                self._json(400, {"error": "request must be JSON with a 'texts' list"})
                return
            texts = [str(t).strip() for t in texts if str(t).strip()]
            if not texts or len(texts) > MAX_BATCH or any(len(t) > MAX_CHARS for t in texts):
                self._json(400, {"error": f"give between 1 and {MAX_BATCH} enquiries, each not too long"})
                return
            use_model = bool(data.get("model")) and self.model_available
            use_tagger = bool(data.get("tagger")) and self.tagger_available
            self._json(200, {"rows": build_batch(texts, use_model, use_tagger)})
        elif self.path == "/save_correction":
            status, body = handle_save(self._read_json(MAX_CHARS * 8), self.model_available, self.corrections_path,
                                       self.tagger_available)
            self._json(status, body)
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args) -> None:  # keep the console quiet during a demo
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Browser demo for the travel enquiry extractor.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-model", action="store_true", help="do not load the model (rules only)")
    parser.add_argument("--no-tagger", action="store_true", help="do not load the trained tagger")
    parser.add_argument("--corrections", type=Path, default=CORRECTIONS_FILE,
                        help="file where corrections are saved (default: data/uc2_inquiries/corrections/)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.no_model:
        model_available, model_message = False, "Model switched off (--no-model), so the demo runs rules only."
    else:
        model_available, model_message = check_model()
    if args.no_tagger:
        tagger_available, tagger_message = False, "Trained tagger switched off (--no-tagger)."
    else:
        tagger_available, tagger_message = check_tagger()
    gold = load_gold_sample()
    Handler.model_available = model_available
    Handler.tagger_available = tagger_available
    Handler.corrections_path = args.corrections
    Handler.page = render_page(SAMPLES + [gold], model_available, model_message, tagger_available, tagger_message)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Travel Enquiry Extractor demo running at http://localhost:{args.port}")
    print(model_message)
    print(tagger_message)
    if gold["text"] is None:
        print(f"Note: {GOLD_FILE} not found, so the gold-set sample button is disabled.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
