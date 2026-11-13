"""Hotel rate demo page (standard library only).

Run from the project root:
    python app/uc21_rate_demo.py      (then open http://localhost:8003)

Shows a predicted nightly rate and a 10-90% range for one booking, the same booking in each
month of 2017 as a small chart, and the results on the locked 2017 test set. Those results are
read from models/uc21_rates/metrics.json when the page starts; nothing is typed in by hand.

The server only listens on this computer (127.0.0.1) and logs nothing. The page inserts all text
with textContent or createElement, never as HTML.
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

from uc2_info import INFO_CSS, INFO_JS  # noqa: E402

from src.uc21_rates import predict as P  # noqa: E402
from src.uc21_rates import explain as E  # noqa: E402
from src.uc21_rates.data import MODEL_DIR  # noqa: E402

MAX_BODY = 4096

FEATURE_NAMES = {
    "arrival_week": "Arrival week", "hotel": "Hotel", "reserved_room_type": "Room type",
    "market_segment": "Market segment", "meal": "Meal plan", "adults": "Adults", "arrival_month": "Arrival month",
    "children": "Children", "lead_time": "Lead time", "customer_type": "Customer type",
    "stays_in_week_nights": "Week nights", "stays_in_weekend_nights": "Weekend nights",
    "required_car_parking_spaces": "Parking spaces", "total_of_special_requests": "Special requests",
    "is_repeated_guest": "Repeat guest", "babies": "Babies", "arrival_weekday": "Arrival weekday",
    "distribution_channel": "Channel",
}
MEAL_NAMES = {"BB": "BB - bed and breakfast", "HB": "HB - half board", "FB": "FB - full board", "SC": "SC - no meals"}
CHANNEL_NAMES = {"TA/TO": "TA/TO - travel agent or tour operator", "GDS": "GDS - global distribution system"}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hotel Rate Estimate</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; font-size: 24px; line-height: 1.5; color: #111; background: #fff; margin: 0; }
  main { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 34px; margin: 0 0 8px; }
  h2 { font-size: 26px; margin: 28px 0 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px 20px; }
  label { display: flex; flex-direction: column; font-size: 20px; font-weight: bold; }
  input, select { font: inherit; font-weight: normal; font-size: 22px; padding: 8px 10px; margin-top: 4px;
                  border: 2px solid #888; border-radius: 6px; background: #fff; }
  button { font-size: 20px; padding: 10px 16px; border-radius: 6px; border: 2px solid #555; background: #f2f2f2; cursor: pointer; }
  button.primary { background: #1f4e8c; color: #fff; border-color: #1f4e8c; font-weight: bold; margin-top: 16px; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .big { font-size: 44px; font-weight: bold; }
  .box { border: 2px solid #ccc; border-radius: 6px; padding: 12px 16px; margin-top: 16px; }
  #chart svg { width: 100%; height: auto; max-width: 900px; display: block; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #ccc; vertical-align: top; }
  td.num, th.num { text-align: right; }
  th { background: #eef1f5; }
  .small { font-size: 19px; color: #444; }
  .error { color: #9b1c1c; font-weight: bold; }
__INFO_CSS__
  .why { list-style: none; padding: 0; margin: 8px 0; }
  .why li { display: grid; grid-template-columns: minmax(220px, 38%) 1fr 170px; gap: 12px; align-items: center;
            padding: 6px 0; border-bottom: 1px solid #e3e3e3; }
  .why-track { background: #f0f0f0; border-radius: 4px; height: 18px; }
  .why-bar { display: block; height: 18px; border-radius: 4px; }
  .why-bar.up { background: #c2410c; }
  .why-bar.down { background: #1f4e8c; }
  .why-num { text-align: right; font-weight: bold; }
</style>
</head>
<body>
<main>
  <h1>Hotel Rate Estimate</h1>
  <p class="small">Trained on bookings at two hotels in Portugal (2015-2016 arrivals). Rates in EUR.
  A reference rate from historical data, not a live quote.</p>
  <p id="not-ready" class="error" hidden></p>

  <form id="booking">
    <div class="grid">
      <label>Hotel <select id="hotel"></select></label>
      <label>Arrival date <input id="arrival_date" type="date" required></label>
      <label>Nights <input id="nights" type="number" step="1" value="3" required></label>
      <label>Adults <input id="adults" type="number" step="1" value="2" required></label>
      <label>Children <input id="children" type="number" step="1" value="0" required></label>
      <label>Room type <select id="room_type"></select></label>
      <label>Meal <select id="meal"></select></label>
      <label>Channel <select id="channel"></select></label>
      <label>Lead time (days from booking to arrival) <input id="lead_time" type="number" step="1" value="30" required></label>
    </div>
    <button class="primary" id="estimate" type="submit">Estimate rate</button>
  </form>
  <p id="status" class="small"></p>

  <div id="result" hidden>
    <div class="box">
      <div>Predicted nightly rate<span data-info-slot="rate"></span></div>
      <div class="big" id="rate"></div>
      <div>10-90% range<span data-info-slot="range"></span>: <strong id="range"></strong></div>
    </div>
    <div id="why-box">
      <h2>Why this price?</h2>
      <p class="small">How each input moves this prediction (what-if, not a guarantee): each input is swapped for the
        typical value in the 2015-2016 training bookings and the same model predicts again.</p>
      <ul id="why" class="why"></ul>
      <p id="why-same" class="small"></p>
    </div>
    <h2>Same booking in each month of 2017<span data-info-slot="season"></span></h2>
    <p class="small">Line: predicted rate. Shaded band: 10-90% range. Orange point: the month of your arrival date.</p>
    <div id="chart"></div>
    <table>
      <thead><tr><th>Month</th><th class="num">Rate (EUR)</th><th class="num">Range (EUR)</th></tr></thead>
      <tbody id="months"></tbody>
    </table>
  </div>

  <h2>Results on the locked 2017 test set<span data-info-slot="results"></span></h2>
  <p id="test-info" class="small"></p>
  <table>
    <thead><tr><th>Model</th><th class="num">MAE (EUR)</th><th class="num">MAPE (%)</th></tr></thead>
    <tbody id="overall"></tbody>
  </table>
  <table>
    <thead id="hotel-head"></thead>
    <tbody id="per-hotel"></tbody>
  </table>
  <p id="coverage" class="small"></p>

  <h2>Top drivers<span data-info-slot="drivers"></span></h2>
  <table>
    <thead><tr><th>#</th><th>Input</th><th class="num">Error rise when shuffled (EUR)</th></tr></thead>
    <tbody id="drivers"></tbody>
  </table>
</main>
<script>
__INFO_JS__
const STATE = __STATE__;
const FEATURE_NAMES = __FEATURE_NAMES__;
const LABELS = {"hotel": {}, "room_type": {}, "meal": __MEAL_NAMES__, "channel": __CHANNEL_NAMES__};
const DEFAULTS = {"hotel": "City Hotel", "room_type": "A", "meal": "BB", "channel": "TA/TO"};
const MODEL_LABELS = {"baseline": "Baseline (median)", "ridge": "Ridge", "gradient_boosting": "Gradient boosting"};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const SVG_NS = "http://www.w3.org/2000/svg";
const statusLine = document.getElementById("status");

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (cls) node.className = cls;
  return node;
}
function row(cells) {
  const tr = document.createElement("tr");
  for (const c of cells) tr.appendChild(el("td", c.text, c.cls));
  return tr;
}
function eur(v) { return Number(v).toFixed(2); }

// ---- form ----
for (const key of ["hotel", "room_type", "meal", "channel"]) {
  const select = document.getElementById(key);
  const values = STATE.allowed[key] || [];
  for (const v of values) {
    const option = el("option", LABELS[key][v] || v);
    option.value = v;
    select.appendChild(option);
  }
  if (values.includes(DEFAULTS[key])) select.value = DEFAULTS[key];
}
for (const [key, range] of Object.entries(STATE.limits)) {
  const input = document.getElementById(key);
  if (input) { input.min = range[0]; input.max = range[1]; }
}
const start = new Date(Date.now() + 30 * 86400000);
document.getElementById("arrival_date").value = start.toISOString().slice(0, 10);
if (!STATE.ready) {
  const msg = document.getElementById("not-ready");
  msg.textContent = STATE.message;
  msg.hidden = false;
  document.getElementById("estimate").disabled = true;
}

function booking() {
  const out = {};
  for (const key of ["hotel", "arrival_date", "room_type", "meal", "channel"]) out[key] = document.getElementById(key).value;
  for (const key of ["nights", "adults", "children", "lead_time"]) out[key] = document.getElementById(key).value.trim();
  return out;
}

async function estimate(event) {
  event.preventDefault();
  statusLine.textContent = "Working...";
  statusLine.className = "small";
  const data = booking();
  try {
    const res = await fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"},
                                          body: JSON.stringify(data)});
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || res.statusText);
    showResult(out, Number(data.arrival_date.slice(5, 7)));
    statusLine.textContent = "";
  } catch (err) {
    statusLine.textContent = "Error: " + err.message;
    statusLine.className = "error";
  }
}
document.getElementById("booking").addEventListener("submit", estimate);

function showWhy(rows) {
  const list = document.getElementById("why");
  list.replaceChildren();
  const moved = rows.filter(r => r.change !== 0);
  const same = rows.filter(r => r.change === 0).map(r => r.label);
  const biggest = Math.max(1, ...moved.map(r => Math.abs(r.change)));
  for (const r of moved) {
    const li = el("li");
    li.appendChild(el("span", r.label + ": " + r.value + " (typical " + r.typical + ")", "why-label"));
    const track = el("span", null, "why-track");
    const bar = el("span", null, "why-bar " + (r.change > 0 ? "up" : "down"));
    bar.style.width = Math.max(2, Math.abs(r.change) / biggest * 100) + "%";
    track.appendChild(bar);
    li.appendChild(track);
    li.appendChild(el("span", (r.change > 0 ? "+" : "\u2212") + eur(Math.abs(r.change)) + " EUR", "why-num"));
    list.appendChild(li);
  }
  if (!rows.length) list.appendChild(el("li", "Not available (typical values not saved yet)."));
  document.getElementById("why-same").textContent = same.length ? "Already the typical value: " + same.join(", ") + "." : "";
}

function showResult(out, month) {
  const p = out.prediction;
  showWhy(out.explain || []);
  document.getElementById("rate").textContent = eur(p.rate) + " " + p.currency;
  document.getElementById("range").textContent = eur(p.low) + " to " + eur(p.high) + " " + p.currency;
  drawChart(out.seasonality, month);
  const body = document.getElementById("months");
  body.replaceChildren();
  for (const r of out.seasonality) {
    body.appendChild(row([{text: MONTHS[r.month - 1] + " 2017"}, {text: eur(r.rate), cls: "num"},
                          {text: eur(r.low) + " to " + eur(r.high), cls: "num"}]));
  }
  document.getElementById("result").hidden = false;
}

// ---- seasonality chart: inline SVG built with createElementNS ----
function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1))));
  const n = raw / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}
function drawChart(rows, selectedMonth) {
  const W = 800, H = 340, L = 70, R = 16, T = 34, B = 44;
  let yMin = Math.min(...rows.map(r => r.low)), yMax = Math.max(...rows.map(r => r.high));
  const step = niceStep((yMax - yMin) / 4);
  yMin = Math.max(0, Math.floor(yMin / step) * step);
  yMax = Math.ceil(yMax / step) * step;
  if (yMax <= yMin) yMax = yMin + step;
  const slot = (W - L - R) / 12;
  const x = i => L + (i + 0.5) * slot;
  const y = v => T + (yMax - v) / (yMax - yMin) * (H - T - B);
  const svg = svgEl("svg", {viewBox: "0 0 " + W + " " + H, role: "img",
                            "aria-label": "Predicted nightly rate for each month of 2017 with the 10-90% range"});
  svg.appendChild(svgEl("text", {x: 8, y: 20, "font-size": 17, fill: "#444"}, "EUR per night"));
  for (let v = yMin; v <= yMax + 1e-9; v += step) {
    svg.appendChild(svgEl("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#e5e5e5", "stroke-width": 1}));
    svg.appendChild(svgEl("text", {x: L - 10, y: y(v) + 6, "text-anchor": "end", "font-size": 17, fill: "#444"},
                          String(Math.round(v))));
  }
  const upper = rows.map((r, i) => x(i) + "," + y(r.high));
  const lower = rows.map((r, i) => x(i) + "," + y(r.low)).reverse();
  svg.appendChild(svgEl("polygon", {points: upper.concat(lower).join(" "), fill: "#1f4e8c", "fill-opacity": 0.15}));
  svg.appendChild(svgEl("polyline", {points: rows.map((r, i) => x(i) + "," + y(r.rate)).join(" "), fill: "none",
                                     stroke: "#1f4e8c", "stroke-width": 2.5, "stroke-linejoin": "round"}));
  rows.forEach((r, i) => {
    const selected = r.month === selectedMonth;
    svg.appendChild(svgEl("circle", {cx: x(i), cy: y(r.rate), r: selected ? 8 : 5,
                                     fill: selected ? "#f59e0b" : "#1f4e8c", stroke: "#fff", "stroke-width": 2}));
    svg.appendChild(svgEl("text", {x: x(i), y: H - B + 28, "text-anchor": "middle", "font-size": 17,
                                   fill: selected ? "#111" : "#444", "font-weight": selected ? "bold" : "normal"},
                          MONTHS[i]));
    const hit = svgEl("rect", {x: x(i) - slot / 2, y: T, width: slot, height: H - T - B, fill: "transparent"});
    hit.appendChild(svgEl("title", {}, MONTHS[i] + " 2017: " + eur(r.rate) + " EUR (range " + eur(r.low) +
                                       " to " + eur(r.high) + ")"));
    svg.appendChild(hit);
  });
  document.getElementById("chart").replaceChildren(svg);
}

// ---- results on the locked test set (from metrics.json, read at start-up) ----
function showMetrics(m) {
  if (!m) {
    document.getElementById("test-info").textContent = "No results yet: metrics.json was not found.";
    return;
  }
  const t = m.test_set;
  document.getElementById("test-info").textContent = t.rows.toLocaleString("en") + " bookings with arrivals from " +
    t.first_arrival + " to " + t.last_arrival + ". Locked before scoring (SHA-256 " + t.sha256.slice(0, 12) +
    "...) and scored once.";
  const overall = document.getElementById("overall");
  for (const name of Object.keys(MODEL_LABELS)) {
    overall.appendChild(row([{text: MODEL_LABELS[name]}, {text: eur(m.overall[name].mae), cls: "num"},
                             {text: Number(m.overall[name].mape).toFixed(1), cls: "num"}]));
  }
  const hotels = Object.keys(m.per_hotel);
  const head = document.createElement("tr");
  head.appendChild(el("th", "MAE (EUR) by hotel"));
  for (const h of hotels) head.appendChild(el("th", h + " (" + m.per_hotel[h].rows.toLocaleString("en") + ")", "num"));
  document.getElementById("hotel-head").appendChild(head);
  const perHotel = document.getElementById("per-hotel");
  for (const name of Object.keys(MODEL_LABELS)) {
    perHotel.appendChild(row([{text: MODEL_LABELS[name]}].concat(
      hotels.map(h => ({text: eur(m.per_hotel[h][name].mae), cls: "num"})))));
  }
  const cov = m.overall.range_10_90.coverage * 100;
  const parts = hotels.map(h => h + " " + (m.per_hotel[h].range_10_90.coverage * 100).toFixed(1) + "%");
  let text = "10-90% range from the quantile models alone: the actual rate was inside it for " +
    cov.toFixed(1) + "% of test bookings (" + parts.join(", ") + "). The target is 80%.";
  if (m.calibrated) {
    const c = m.calibrated;
    const cparts = hotels.map(h => h + " " + (c.per_hotel[h].coverage * 100).toFixed(1) + "%");
    text += " The range shown above is calibrated (conformal, widened by " + c.q.toFixed(2) + " EUR at each end): " +
      "it held " + (c.overall.coverage * 100).toFixed(1) + "% (" + cparts.join(", ") + "), still below 80%.";
  }
  document.getElementById("coverage").textContent = text;
  const drivers = document.getElementById("drivers");
  m.top_drivers.forEach((d, i) => {
    drivers.appendChild(row([{text: String(i + 1)}, {text: (FEATURE_NAMES[d.feature] || d.feature) + " (" + d.feature + ")"},
                             {text: eur(d.mae_increase), cls: "num"}]));
  });
}
showMetrics(STATE.metrics);
</script>
</body>
</html>
"""


def _script_json(data) -> str:
    # json.dumps output is safe inside a script block once "</" is escaped
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def load_state(model_dir=MODEL_DIR) -> dict:
    """Read the allowed values, limits and test-set results once, at start-up."""
    state = {"ready": False, "message": "", "allowed": {}, "limits": {}, "metrics": None}
    try:
        P.load_models(model_dir)
        state["allowed"] = P.allowed_values(model_dir)
        state["limits"] = P.numeric_limits(model_dir)
        state["ready"] = True
    except FileNotFoundError:
        state["message"] = "Models not found. Train them first: python -m src.uc21_rates.train"
    metrics_path = Path(model_dir) / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        state["metrics"] = {"test_set": m["test_set"], "overall": m["overall"], "per_hotel": m["per_hotel"],
                            "top_drivers": m["top_drivers"][:8], "scored_on": m["scored_on"]}
        calibrated = Path(model_dir) / "calibration_metrics.json"
        if calibrated.exists():  # written once by calibrate.py
            c = json.loads(calibrated.read_text(encoding="utf-8"))
            state["metrics"]["calibrated"] = {"overall": c["overall"], "per_hotel": c["per_hotel"], "q": c["q"]}
    return state


def info_texts(metrics: dict | None) -> dict:
    range_text = "A range where most similar bookings fell; not a guarantee."
    if metrics and metrics.get("calibrated"):
        cov = metrics["calibrated"]["overall"]["coverage"] * 100
        range_text += (" It is calibrated to hold about 80% of late-2016 bookings. On the locked 2017 test set the "
                       f"actual rate was inside it for {cov:.1f}% of bookings.")
    elif metrics:
        cov = metrics["overall"]["range_10_90"]["coverage"] * 100
        range_text += f" On the locked 2017 test set the actual rate was inside it for {cov:.1f}% of bookings."
    return {
        "rate": "The gradient boosting model's estimate of the average nightly rate for this booking, in EUR. "
                "It is based on 2015-2016 bookings at two hotels in Portugal, so it is a reference, not a live quote.",
        "range": range_text,
        "season": "The same booking moved to the 15th of each month of 2017. The shaded band is the 10-90% range.",
        "results": "Scored once on 2017 bookings that the models never saw while training. MAE is the average "
                   "error in EUR and MAPE the average error in percent. Lower is better. The baseline is the "
                   "median rate for the same hotel, month and room type in the training data.",
        "drivers": "How much the average error rises (EUR) when one input is shuffled on test bookings. "
                   "A bigger number means the model relies more on that input.",
    }


def render_page(state: dict) -> bytes:
    page = PAGE.replace("__STATE__", _script_json(state))
    page = page.replace("__FEATURE_NAMES__", _script_json(FEATURE_NAMES))
    page = page.replace("__MEAL_NAMES__", _script_json(MEAL_NAMES))
    page = page.replace("__CHANNEL_NAMES__", _script_json(CHANNEL_NAMES))
    info_js = INFO_JS.replace("__INFO__", _script_json(info_texts(state["metrics"])))
    page = page.replace("__INFO_CSS__", INFO_CSS).replace("__INFO_JS__", info_js)
    return page.encode("utf-8")


def handle_predict(data, model_dir=MODEL_DIR) -> tuple[int, dict]:
    if not isinstance(data, dict):
        return 400, {"error": "send the booking as a JSON object"}
    try:
        return 200, {"prediction": P.predict_rate(data, model_dir), "seasonality": P.seasonality(data, model_dir),
                     "explain": E.explain(data, model_dir)}
    except ValueError as err:
        return 400, {"error": str(err)}


class Handler(BaseHTTPRequestHandler):
    page: bytes = b""
    ready: bool = False
    model_dir = MODEL_DIR

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
        if self.path != "/predict":
            self._json(404, {"error": "not found"})
            return
        if not self.ready:
            self._json(503, {"error": "models not found; train them first"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8")) if 0 < length <= MAX_BODY else None
        except ValueError:
            data = None
        status, out = handle_predict(data, self.model_dir)
        self._json(status, out)

    def log_message(self, fmt, *args) -> None:  # nothing is logged
        pass


def make_server(port: int, model_dir=MODEL_DIR) -> ThreadingHTTPServer:
    state = load_state(model_dir)
    Handler.page = render_page(state)
    Handler.ready = state["ready"]
    Handler.model_dir = model_dir
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hotel rate demo page.")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    server = make_server(args.port)
    print(f"Hotel rate demo running at http://localhost:{args.port}")
    if not Handler.ready:
        print("Models not found. Train them first: python -m src.uc21_rates.train")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
