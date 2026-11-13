"""Home page that links the project's tools (standard library only).

Run from the project root:
    python app/home.py            (then open http://localhost:8004)

One card per tool, with a one-line description, a live status dot (checked from the browser),
an Open button and one or two headline numbers. The numbers are read from the project's results
files each time the page loads, never typed in here; a missing file shows "not available".
The server only listens on this computer (127.0.0.1). The page inserts all text with
textContent, never as HTML.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_OPENAPI = "http://127.0.0.1:8001/openapi.json"
NOT_AVAILABLE = "not available"

USE_CASES = [
    {"name": "UC 2.2 Inquiry Extractor", "status": "built"},
    {"name": "UC 2.1 Hotel Rate Prediction", "status": "first version"},
    {"name": "C2", "status": "planned"},
]
CARDS = [
    {"key": "extractor", "title": "2.2 Inquiry Extractor", "port": 8000, "open": "/", "check": "/",
     "description": "Turns a travel enquiry in English, French or German into a checked client profile."},
    {"key": "highlighter", "title": "Enquiry Highlighter", "port": 8002, "open": "/", "check": "/",
     "description": "Shows the enquiry with every extracted value highlighted in the colour of its field."},
    {"key": "rates", "title": "Hotel Rate Prediction", "port": 8003, "open": "/", "check": "/",
     "description": "Estimates a reference nightly rate and a price range for a hotel booking."},
    {"key": "api", "title": "API docs", "port": 8001, "open": "/docs", "check": "/health",
     "description": "The local API for both use cases, with interactive documentation."},
]


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(label: str, read) -> dict:
    """One headline number. Anything missing or unreadable shows as not available."""
    try:
        value = read()
    except (TypeError, KeyError, IndexError, AttributeError, ValueError):
        value = None
    return {"label": label, "value": value if value else NOT_AVAILABLE}


def _fair_number(root: Path):
    try:
        text = (root / "docs" / "uc2_results.md").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"only fair number is the first held-out run\*\* \((\d\.\d\d) / (\d\.\d\d) / (\d\.\d\d)\)", text)
    return f"en {m.group(1)} / fr {m.group(2)} / de {m.group(3)}" if m else None


def _endpoint_count(url: str):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            spec = json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return str(sum(len(methods) for methods in spec.get("paths", {}).values()))


def headline_numbers(root: Path = ROOT, api_url: str = API_OPENAPI) -> dict[str, list[dict]]:
    comparison = _json(root / "models" / "uc2_crf" / "comparison_v2.json")
    tagger = _json(root / "models" / "uc2_crf" / "crf_tagger.json")
    rates = _json(root / "models" / "uc21_rates" / "metrics.json")
    calibrated = _json(root / "models" / "uc21_rates" / "calibration_metrics.json")
    return {
        "extractor": [
            _number("Held-out set 1, rules only, first run (the real number), F1", lambda: _fair_number(root)),
            _number("Held-out set 1 field errors (optimistic): rules only / rules + new tagger",
                    lambda: f"{comparison['heldout']['errors']['rules only']} / {comparison['heldout']['errors']['rules + v2']}"),
        ],
        "highlighter": [
            _number("Trained tagger, development F1",
                    lambda: " / ".join(f"{lang} {tagger['dev_report'][lang]['ALL']['f1']:.2f}" for lang in ("en", "fr", "de"))),
        ],
        "rates": [
            _number("Locked 2017 test set, MAE: gradient boosting / baseline",
                    lambda: f"{rates['overall']['gradient_boosting']['mae']:.2f} / {rates['overall']['baseline']['mae']:.2f} EUR"),
            _number("Calibrated 10-90% range held (target 80%)",
                    lambda: f"{calibrated['overall']['coverage'] * 100:.1f}%"),
        ],
        "api": [
            _number("Endpoints, read from the running API", lambda: _endpoint_count(api_url)),
        ],
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kenexoft AI/ML</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; font-size: 20px; line-height: 1.45; color: #1a1a1a;
         background: #ffffff; margin: 0; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px 20px 40px; }
  h1 { font-size: 32px; margin: 0 0 6px; }
  h2 { font-size: 24px; margin: 0; }
  .small { font-size: 17px; color: #555; }
  .strip { display: flex; flex-wrap: wrap; gap: 10px; list-style: none; padding: 0; margin: 16px 0 22px; }
  .strip li { padding: 6px 14px; border-radius: 6px; border: 2px solid #1f4e8c; background: #f5f8fc; }
  .strip strong { margin-right: 6px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }
  .card { border: 2px solid #ccc; border-radius: 8px; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; }
  .title { display: flex; align-items: center; gap: 10px; }
  .dot { width: 16px; height: 16px; border-radius: 50%; background: #9a9a9a; flex: none; }
  .dot.up { background: #1d8a3a; }
  .state { font-size: 16px; color: #555; }
  dl { margin: 0; }
  dt { font-size: 16px; color: #555; }
  dd { margin: 0 0 8px; font-weight: bold; font-size: 22px; }
  dd.missing { font-weight: normal; color: #9a9a9a; font-size: 18px; }
  a.button { align-self: flex-start; font-size: 18px; padding: 10px 18px; border-radius: 6px; text-decoration: none;
             background: #1f4e8c; color: #fff; font-weight: bold; margin-top: auto; }
  footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid #ccc; color: #555; font-size: 17px; }
</style>
</head>
<body>
<main>
  <h1>Kenexoft AI/ML</h1>
  <p class="small">The tools of this internship project. Everything runs on this computer.</p>
  <ul id="strip" class="strip" aria-label="Use cases"></ul>
  <div id="cards" class="cards"></div>
  <footer>Numbers are read from the project's results files each time this page loads. "Optimistic" means the
    set was seen before; the only real number for the extractor is its first held-out run. A grey dot means the
    tool is not running.</footer>
</main>
<script>
const USE_CASES = __USE_CASES__;
const CARDS = __CARDS__;
const NUMBERS = __NUMBERS__;

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

const strip = document.getElementById("strip");
for (const u of USE_CASES) {
  const li = el("li");
  li.appendChild(el("strong", u.name));
  li.appendChild(document.createTextNode(u.status));
  strip.appendChild(li);
}

function base(card) { return location.protocol + "//" + location.hostname + ":" + card.port; }

const checks = [];
const box = document.getElementById("cards");
for (const card of CARDS) {
  const section = el("section", null, "card");
  const title = el("div", null, "title");
  const dot = el("span", null, "dot");
  dot.setAttribute("aria-hidden", "true");
  const state = el("span", "checking\\u2026", "state");
  title.appendChild(dot);
  title.appendChild(el("h2", card.title));
  title.appendChild(state);
  section.appendChild(title);
  section.appendChild(el("p", card.description));
  const list = el("dl");
  for (const n of NUMBERS[card.key] || []) {
    list.appendChild(el("dt", n.label));
    list.appendChild(el("dd", n.value, n.value === "not available" ? "missing" : ""));
  }
  section.appendChild(list);
  const open = el("a", "Open", "button");
  open.href = base(card) + card.open;
  open.target = "_blank";
  open.rel = "noopener";
  section.appendChild(open);
  box.appendChild(section);
  checks.push({card, dot, state});
}

// A server counts as up if it answers at all. "no-cors" lets the browser ask without reading the reply.
async function answers(url) {
  const stop = new AbortController();
  const timer = setTimeout(() => stop.abort(), 3000);
  try {
    await fetch(url, {mode: "no-cors", cache: "no-store", signal: stop.signal});
    return true;
  } catch (err) {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
async function refresh() {
  for (const c of checks) {
    const up = await answers(base(c.card) + c.card.check);
    c.dot.className = "dot" + (up ? " up" : "");
    c.state.textContent = up ? "running" : "not running";
  }
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


def _script_json(data) -> str:
    # json.dumps output is safe inside a script block once "</" is escaped
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_page(numbers: dict) -> bytes:
    page = PAGE.replace("__USE_CASES__", _script_json(USE_CASES))
    page = page.replace("__CARDS__", _script_json(CARDS))
    page = page.replace("__NUMBERS__", _script_json(numbers))
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body, status, kind = render_page(headline_numbers()), 200, "text/html; charset=utf-8"
        else:
            body, status, kind = b'{"error": "not found"}', 404, "application/json"
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:  # keep the console quiet
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Home page for the project's tools.")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Home page running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
