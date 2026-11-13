"""Highlight page for the travel enquiry extractor (standard library only).

Run from the project root:
    python app/uc2_demo_highlight.py      (then open http://localhost:8002)

Shows the enquiry with each extracted value highlighted in the colour of its field, using the
text positions stored in the profile's evidence, with a colour legend and a table of values.
The rules run first and the trained tagger fills what they left empty, as in rules_tagger mode.

The server only listens on this computer (127.0.0.1), and nothing is saved or logged. The page
inserts enquiry text with textContent, never as HTML, so pasted text cannot run as code.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from uc2_info import INFO_CSS, info_script  # noqa: E402
from uc2_web_demo import MAX_CHARS, ROWS, SAMPLES, _script_json, build_rows, load_gold_sample  # noqa: E402

from src.uc2_extractor.extract import extract_profile  # noqa: E402
from src.uc2_extractor.tagger import check_tagger  # noqa: E402

# field -> (name in the legend, colour)
COLOURS = {
    "destination": ("Destination", "#bfdbfe"),
    "dates": ("Dates", "#fde68a"),
    "travellers": ("Travellers", "#bbf7d0"),
    "budget": ("Budget", "#fecaca"),
    "accommodation": ("Accommodation", "#ddd6fe"),
    "flights": ("Flights", "#a5f3fc"),
    "interests": ("Interests", "#fed7aa"),
    "dietary": ("Diet", "#d9f99d"),
    "occasion": ("Occasion", "#fbcfe8"),
    "special_requests": ("Special requests", "#d6d3d1"),
}
ORDER = {field: i for i, field in enumerate(COLOURS)}


def highlight_segments(text: str, evidence) -> list[dict]:
    """Cut the text at every evidence edge. Joined together, the pieces are exactly the text.

    Each piece lists the fields whose evidence covers it, so overlapping evidence is kept.
    Evidence that does not match the text at its position is skipped.
    """
    spans = [(e.start, e.end, e.field) for e in evidence
             if e.field in COLOURS and 0 <= e.start < e.end <= len(text) and text[e.start:e.end] == e.text]
    cuts = sorted({0, len(text)} | {s[0] for s in spans} | {s[1] for s in spans})
    pieces: list[dict] = []
    for a, b in zip(cuts, cuts[1:]):
        fields = sorted({f for start, end, f in spans if start <= a and b <= end}, key=ORDER.get)
        if pieces and pieces[-1]["fields"] == fields:
            pieces[-1]["text"] += text[a:b]
        else:
            pieces.append({"text": text[a:b], "fields": fields})
    return pieces


def tag_text(item: dict) -> str:
    source, confidence = item["source"], item["confidence"]
    names = {"rule": "rule", "crf": "crf", "rule+crf": "rule + crf", "model": "model"}
    return f"{names[source]} {confidence:.2f}" if confidence is not None else names[source]


def build_response(text: str, use_tagger: bool) -> dict:
    profile = extract_profile(text, tagger=use_tagger)
    rows = []
    for (key, label), row in zip(ROWS, build_rows(profile)):
        for item in row["items"]:
            rows.append({"value": item["text"], "field": COLOURS.get(key, (label,))[0], "key": key,
                         "tag": tag_text(item)})
    return {"segments": highlight_segments(text, profile.evidence), "rows": rows}


def sample_list() -> list[dict]:
    gold = load_gold_sample()
    gold["label"] = "English: honeymoon"
    return [{"label": s["label"], "text": s["text"]} for s in SAMPLES[:3]] + [gold, {
        "label": SAMPLES[3]["label"], "text": SAMPLES[3]["text"]}]


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enquiry Highlighter</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; font-size: 24px; line-height: 1.5; color: #111; background: #fff; margin: 0; }
  main { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 34px; margin: 0 0 12px; }
  h2 { font-size: 26px; margin: 26px 0 8px; }
  textarea { width: 100%; box-sizing: border-box; height: 150px; font: inherit; padding: 12px;
             border: 2px solid #888; border-radius: 6px; }
  .buttons { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
  button { font-size: 20px; padding: 10px 16px; border-radius: 6px; border: 2px solid #555; background: #f2f2f2; cursor: pointer; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.primary { background: #1f4e8c; color: #fff; border-color: #1f4e8c; font-weight: bold; }
  #marked { white-space: pre-wrap; font-size: 26px; line-height: 1.8; padding: 16px; border: 2px solid #ccc; border-radius: 6px; }
  mark { color: #111; padding: 2px 4px; border-radius: 4px; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; list-style: none; padding: 0; margin: 12px 0; }
  .legend li { padding: 3px 12px; border-radius: 4px; font-size: 20px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #ccc; vertical-align: top; }
  th { background: #eef1f5; }
  .swatch { display: inline-block; width: 18px; height: 18px; border-radius: 3px; margin-right: 8px; vertical-align: -2px; background: #e5e5e5; }
  .crf { font-weight: bold; color: #14532d; }
  .small { font-size: 19px; color: #444; }
__INFO_CSS__
</style>
</head>
<body>
<main>
  <h1>Enquiry Highlighter</h1>
  <textarea id="enquiry" maxlength="20000" aria-label="Enquiry text"></textarea>
  <div class="buttons">
    <button class="primary" id="extract">Extract</button>
    <span id="samples" class="buttons" style="margin:0"></span>
  </div>
  <p class="small"><span id="status"></span><span data-info-slot="mode"></span></p>
  <ul id="legend" class="legend"></ul>
  <div id="result" hidden>
    <h2>Enquiry<span data-info-slot="highlight"></span></h2>
    <div id="marked"></div>
    <h2>Values</h2>
    <p class="small">Tags: rule<span data-info-slot="tag_rule"></span> &nbsp; crf 0.xx<span data-info-slot="tag_crf"></span></p>
    <table>
      <thead><tr><th>Value</th><th>Field</th><th>Tag</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</main>
<script>
__INFO_JS__
const SAMPLES = __SAMPLES__;
const COLOURS = __COLOURS__;
const MODE = __MODE__;
const box = document.getElementById("enquiry");
const statusLine = document.getElementById("status");

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

statusLine.textContent = MODE;
const legend = document.getElementById("legend");
const legendTitle = el("li", "Colours");
legendTitle.appendChild(infoButton("highlight"));
legend.appendChild(legendTitle);
for (const [name, colour] of Object.values(COLOURS)) {
  const li = el("li", name);
  li.style.background = colour;
  legend.appendChild(li);
}

function show(data) {
  const marked = document.getElementById("marked");
  marked.replaceChildren();
  for (const piece of data.segments) {
    if (!piece.fields.length) { marked.appendChild(document.createTextNode(piece.text)); continue; }
    const mark = el("mark", piece.text);
    mark.style.background = COLOURS[piece.fields[0]][1];
    if (piece.fields.length > 1) mark.style.borderBottom = "5px solid " + COLOURS[piece.fields[1]][1];
    mark.title = piece.fields.map(f => COLOURS[f][0]).join(", ");
    marked.appendChild(mark);
  }
  const rows = document.getElementById("rows");
  rows.replaceChildren();
  for (const r of data.rows) {
    const tr = el("tr");
    tr.appendChild(el("td", r.value));
    const field = el("td");
    const swatch = el("span", null, "swatch");
    if (COLOURS[r.key]) swatch.style.background = COLOURS[r.key][1];
    field.appendChild(swatch);
    field.appendChild(document.createTextNode(r.field));
    for (const key of infoKeys(r.field, r.value)) field.appendChild(infoButton(key));
    tr.appendChild(field);
    tr.appendChild(el("td", r.tag, r.tag.includes("crf") ? "crf" : ""));
    rows.appendChild(tr);
  }
  document.getElementById("result").hidden = false;
}

async function extract() {
  const text = box.value.trim();
  if (!text) { statusLine.textContent = "Please paste an enquiry first."; return; }
  statusLine.textContent = "Extracting\\u2026";
  try {
    const res = await fetch("/extract", {method: "POST", headers: {"Content-Type": "application/json"},
                                          body: JSON.stringify({text})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    show(data);
    statusLine.textContent = MODE;
  } catch (err) {
    statusLine.textContent = "Error: " + err.message;
  }
}

const sampleSpan = document.getElementById("samples");
for (const s of SAMPLES) {
  const b = el("button", s.label);
  if (s.text === null) b.disabled = true;
  else b.addEventListener("click", () => { box.value = s.text; extract(); });
  sampleSpan.appendChild(b);
}
document.getElementById("extract").addEventListener("click", extract);
</script>
</body>
</html>
"""


def render_page(samples: list[dict], mode: str) -> bytes:
    page = PAGE.replace("__SAMPLES__", _script_json(samples))
    page = page.replace("__COLOURS__", _script_json(COLOURS))
    page = page.replace("__MODE__", _script_json(mode))
    page = page.replace("__INFO_CSS__", INFO_CSS).replace("__INFO_JS__", info_script(_script_json))
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    page: bytes = b""
    use_tagger: bool = False

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
            data = json.loads(self.rfile.read(length).decode("utf-8")) if 0 < length <= MAX_CHARS * 4 else None
        except ValueError:
            data = None
        text = str(data.get("text", "")).strip() if isinstance(data, dict) else ""
        if not text or len(text) > MAX_CHARS:
            self._json(400, {"error": "enquiry is empty or too long"})
            return
        self._json(200, build_response(text, self.use_tagger))

    def log_message(self, fmt, *args) -> None:  # enquiry text is never logged
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Highlight page for the travel enquiry extractor.")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    available, message = check_tagger()
    mode = "Mode: rules + trained tagger" if available else "Mode: rules only (trained tagger not available)"
    Handler.use_tagger = available
    Handler.page = render_page(sample_list(), mode)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Enquiry highlighter running at http://localhost:{args.port}")
    print(message)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
