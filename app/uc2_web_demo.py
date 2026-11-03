"""Small browser demo for the travel enquiry extractor (standard library only).

Run from the project root:
    python app/uc2_web_demo.py            (then open http://localhost:8000)
    python app/uc2_web_demo.py --port 8080

The server only listens on this computer (127.0.0.1). The page inserts all
results with textContent, never as HTML, so pasted text cannot run as code.
The gold-set sample is read from data/uc2_inquiries/gold/ at start-up and is
not stored in this file; if that file is missing the button is disabled.
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

SAMPLES = [
    {"label": "English: anniversary",
     "text": "Anniversary trip for two, late February, we love gardens, no early mornings"},
    {"label": "French: Rajasthan",
     "text": "Nous sommes deux adultes et deux enfants, nous voulons visiter le Rajasthan pendant dix nuits"},
    {"label": "German: 3/4 dates",
     "text": "Wir reisen vom 3/4 bis 10/4, zwei Erwachsene, vegetarisch"},
]

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


def describe(p: ClientProfile) -> dict[str, str]:
    """One readable string per row; empty string means nothing found."""
    out = {key: "" for key, _ in ROWS}
    if p.language:
        out["language"] = f"{LANGUAGE_NAMES[p.language]} ({p.language})"
    out["destination"] = ", ".join(p.destination)

    d = p.dates
    bits = []
    if d.start and d.end:
        bits.append(f"{_date(d.start)} to {_date(d.end)}")
    elif d.start:
        bits.append(f"from {_date(d.start)}")
    if d.duration:
        bits.append(d.duration.to_str())
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

    for key in ("accommodation", "flights", "interests", "dietary", "occasion", "special_requests"):
        out[key] = ", ".join(getattr(p, key))
    return out


def build_response(text: str) -> dict:
    profile = extract_profile(text)
    values = describe(profile)
    evidence: dict[str, list[str]] = {}
    for ev in profile.evidence:
        evidence.setdefault(ev.field, []).append(ev.text)
    rows = [
        {"label": label, "value": values[key], "evidence": evidence.get(key, [])}
        for key, label in ROWS
    ]
    flags = [{"field": f.field.replace("_", " "), "kind": f.kind, "message": f.message} for f in profile.flags]
    return {"rows": rows, "flags": flags, "profile": profile.model_dump(mode="json")}


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
  main { max-width: 1100px; margin: 0 auto; padding: 24px 20px 40px; }
  h1 { font-size: 32px; margin: 0 0 16px; }
  textarea { width: 100%; box-sizing: border-box; height: 170px; font-size: 20px; padding: 12px;
             border: 2px solid #888; border-radius: 6px; font-family: inherit; }
  .buttons { margin: 12px 0 20px; display: flex; flex-wrap: wrap; gap: 10px; }
  button { font-size: 18px; padding: 10px 16px; border-radius: 6px; border: 2px solid #555;
           background: #f2f2f2; cursor: pointer; }
  button.primary { background: #1f4e8c; color: #fff; border-color: #1f4e8c; font-weight: bold; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid #ccc; }
  th { background: #eef1f5; }
  td.field { font-weight: bold; width: 18%; white-space: nowrap; }
  td.value { width: 42%; }
  td.source { color: #444; font-style: italic; }
  .empty { color: #9a9a9a; font-style: normal; }
  .flags { margin-top: 22px; padding: 14px 18px; background: #fff4d6; border: 2px solid #e0a100;
           border-radius: 6px; }
  .flags h2 { font-size: 22px; margin: 0 0 8px; color: #7a5200; }
  .flags ul { margin: 0; padding-left: 24px; }
  .kind { font-weight: bold; color: #7a5200; }
  pre { background: #f6f6f6; border: 1px solid #ccc; padding: 12px; font-size: 15px;
        overflow-x: auto; white-space: pre-wrap; }
  .error { color: #a00000; font-weight: bold; }
  footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid #ccc; color: #555; font-size: 17px; }
</style>
</head>
<body>
<main>
  <h1>Travel Enquiry Extractor (English, French, German)</h1>
  <label for="enquiry">Paste an enquiry:</label>
  <textarea id="enquiry" maxlength="20000"></textarea>
  <div class="buttons">
    <button class="primary" id="extract">Extract</button>
    <span id="samples"></span>
  </div>
  <p id="status"></p>
  <div id="results" hidden>
    <table>
      <thead><tr><th>Field</th><th>Value</th><th>Taken from the text</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="flags" id="flagbox">
      <h2>Check these</h2>
      <ul id="flags"></ul>
    </div>
    <p><button id="toggle-json">Show raw JSON</button></p>
    <pre id="json" hidden></pre>
  </div>
  <footer>Rules-based first version. Scored on our own synthetic set. Budget bands are placeholders.</footer>
</main>
<script>
const SAMPLES = __SAMPLES__;
const box = document.getElementById("enquiry");
const statusLine = document.getElementById("status");

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

const sampleSpan = document.getElementById("samples");
for (const s of SAMPLES) {
  const b = el("button", s.label);
  if (s.text === null) {
    b.disabled = true;
    b.title = "Gold set file not found in data/uc2_inquiries/gold/";
  } else {
    b.addEventListener("click", () => { box.value = s.text; extract(); });
  }
  sampleSpan.appendChild(b);
  sampleSpan.appendChild(document.createTextNode(" "));
}

function show(data) {
  const tbody = document.getElementById("rows");
  tbody.replaceChildren();
  for (const r of data.rows) {
    const tr = el("tr");
    tr.appendChild(el("td", r.label, "field"));
    tr.appendChild(r.value ? el("td", r.value, "value") : el("td", "\\u2014", "value empty"));
    const src = r.evidence.length ? r.evidence.map(t => "\\u201c" + t + "\\u201d").join("  \\u00b7  ") : "\\u2014";
    tr.appendChild(el("td", src, r.evidence.length ? "source" : "source empty"));
    tbody.appendChild(tr);
  }
  const list = document.getElementById("flags");
  list.replaceChildren();
  if (data.flags.length === 0) {
    list.appendChild(el("li", "Nothing flagged."));
  }
  for (const f of data.flags) {
    const li = el("li");
    li.appendChild(el("span", f.kind + ": ", "kind"));
    li.appendChild(document.createTextNode(f.field + " \\u2013 " + f.message));
    list.appendChild(li);
  }
  document.getElementById("json").textContent = JSON.stringify(data.profile, null, 2);
  document.getElementById("results").hidden = false;
}

async function extract() {
  const text = box.value.trim();
  if (!text) { statusLine.textContent = "Please paste an enquiry first."; return; }
  statusLine.textContent = "Extracting\\u2026";
  statusLine.className = "";
  try {
    const res = await fetch("/extract", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    show(data);
    statusLine.textContent = "";
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


def render_page(samples: list[dict]) -> bytes:
    # json.dumps output is safe inside a script block once "</" is escaped
    data = json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__SAMPLES__", data).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    page: bytes = b""

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, self.page, "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/extract":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_CHARS * 4:
            self._json(400, {"error": "enquiry is empty or too long"})
            return
        try:
            text = json.loads(self.rfile.read(length).decode("utf-8")).get("text", "")
        except (ValueError, AttributeError):
            self._json(400, {"error": "request must be JSON with a 'text' field"})
            return
        text = str(text).strip()
        if not text or len(text) > MAX_CHARS:
            self._json(400, {"error": "enquiry is empty or too long"})
            return
        self._json(200, build_response(text))

    def log_message(self, fmt, *args) -> None:  # keep the console quiet during a demo
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Browser demo for the travel enquiry extractor.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gold = load_gold_sample()
    Handler.page = render_page(SAMPLES + [gold])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Travel Enquiry Extractor demo running at http://localhost:{args.port}")
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
