"""Small browser demo for the travel enquiry extractor (standard library only).

Run from the project root:
    python app/uc2_web_demo.py            (then open http://localhost:8000)
    python app/uc2_web_demo.py --port 8080

Features: one enquiry or several (one per line), rules only or rules plus the
optional model layer, a rule/model tag with confidence on every value, and the
evaluation scores.

The server only listens on this computer (127.0.0.1). The page inserts all
results with textContent, never as HTML, so pasted text cannot run as code.
The gold-set sample is read from data/uc2_inquiries/gold/ at start-up and is
not stored in this file; if that file is missing the button is disabled. If
the model files are missing, the demo runs rules only and says so.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.uc2_extractor.extract import extract_profile  # noqa: E402
from src.uc2_extractor.schema import ClientProfile, PartialDate  # noqa: E402

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
            ["Gold set (60 enquiries), rules", "0.96", "0.98", "0.98"],
        ],
    },
    "next": "The next real number will come from a second held-out set written by colleagues, scored once at the end.",
}

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
            items = [{"text": sections[key], "source": "rule", "confidence": None}] if sections[key] else []
        rows.append({"label": label, "items": items, "evidence": evidence.get(key, [])})
    return rows


def build_response(text: str, use_model: bool) -> dict:
    profile = extract_profile(text, model=use_model)
    flags = [{"field": f.field.replace("_", " "), "kind": f.kind, "message": f.message} for f in profile.flags]
    return {"rows": build_rows(profile), "flags": flags, "profile": profile.model_dump(mode="json")}


def build_batch(texts: list[str], use_model: bool) -> list[dict]:
    """A short summary row per enquiry for the several-enquiries table."""
    out = []
    for i, text in enumerate(texts, 1):
        p = extract_profile(text, model=use_model)
        sections = describe_sections(p)
        source = {(x.field, x.value): x for x in p.provenance}

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
            "dates": sections["dates"],
            "travellers": sections["travellers"],
            "budget": sections["budget"],
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
  footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid #ccc; color: #555; font-size: 17px; }
</style>
</head>
<body>
<main>
  <h1>Travel Enquiry Extractor (English, French, German)</h1>
  <p id="model-status" class="notice"></p>

  <div class="controls">
    <span><strong>Input:</strong>
      <label><input type="radio" name="input" value="one" checked> one enquiry</label>
      <label><input type="radio" name="input" value="many"> several enquiries, one per line</label>
    </span>
    <span><strong>Mode:</strong>
      <label><input type="radio" name="mode" value="rules" checked> rules only</label>
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
    <table>
      <thead><tr><th>Field</th><th>Value</th><th>Taken from the text</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="flags">
      <h3>Check these</h3>
      <ul id="flags"></ul>
    </div>
    <p><button id="toggle-json">Show raw JSON</button></p>
    <pre id="json" hidden></pre>
  </div>

  <div id="batch" hidden>
    <table>
      <thead><tr><th>#</th><th>Lang.</th><th>Destination</th><th>Dates</th><th>Travellers</th>
        <th>Budget</th><th>Other details</th><th>Flags</th></tr></thead>
      <tbody id="batch-rows"></tbody>
    </table>
  </div>

  <section class="scores">
    <h2>How well it works</h2>
    <p class="small">F1 score (1.00 would be perfect), all fields pooled, exact match.</p>
    <div id="scores"></div>
  </section>

  <footer>Rules-based extractor with an optional model layer. Scored on our own synthetic sets. Budget bands are placeholders.</footer>
</main>
<script>
const SAMPLES = __SAMPLES__;
const SCORES = __SCORES__;
const MODEL = __MODEL__;
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
status.textContent = MODEL.message;
if (!MODEL.available) {
  status.className = "notice warn";
  const radio = document.querySelector('#model-choice input');
  radio.disabled = true;
  document.getElementById("model-choice").title = "Model files not found: rules only";
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
  scoreBox.appendChild(el("h3", part.title));
  scoreBox.appendChild(el("p", part.note, "small"));
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
    tr.appendChild(el("td", r.label, "field"));
    tr.appendChild(itemList(r.items));
    const src = el("td", null, r.evidence.length ? "source" : "source empty");
    if (!r.evidence.length) src.textContent = "\\u2014";
    r.evidence.forEach((ev, i) => {
      if (i) src.appendChild(document.createTextNode("  \\u00b7  "));
      src.appendChild(document.createTextNode("\\u201c" + ev.text + "\\u201d"));
      if (ev.source === "model") src.appendChild(el("span", "model", "tag model"));
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
  document.getElementById("batch").hidden = true;
  document.getElementById("single").hidden = false;
}

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
  statusLine.textContent = "Extracting\\u2026";
  statusLine.className = "";
  try {
    if (choice("input") === "many") {
      const texts = text.split(/\\r?\\n/).map(t => t.trim()).filter(t => t);
      showBatch((await post("/extract_batch", {texts, model: useModel})).rows);
    } else {
      showSingle(await post("/extract", {text, model: useModel}));
    }
    statusLine.textContent = "Mode used: " + (useModel ? "rules + model" : "rules only");
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


def render_page(samples: list[dict], model_available: bool, model_message: str) -> bytes:
    page = PAGE.replace("__SAMPLES__", _script_json(samples))
    page = page.replace("__SCORES__", _script_json(SCORES))
    page = page.replace("__MODEL__", _script_json({"available": model_available, "message": model_message}))
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    page: bytes = b""
    model_available: bool = False

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
            self._json(200, build_response(text, use_model))
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
            self._json(200, {"rows": build_batch(texts, use_model)})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args) -> None:  # keep the console quiet during a demo
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Browser demo for the travel enquiry extractor.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-model", action="store_true", help="do not load the model (rules only)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.no_model:
        model_available, model_message = False, "Model switched off (--no-model), so the demo runs rules only."
    else:
        model_available, model_message = check_model()
    gold = load_gold_sample()
    Handler.model_available = model_available
    Handler.page = render_page(SAMPLES + [gold], model_available, model_message)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Travel Enquiry Extractor demo running at http://localhost:{args.port}")
    print(model_message)
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
