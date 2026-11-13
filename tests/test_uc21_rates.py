"""Tests for UC 2.1 hotel rate prediction.

Covers temporal leakage, metric correctness, the baseline fallback, input checks in predict.py,
the derived nights and week number, the seasonality output and the demo page. Tests that need
the data file or the trained models skip when those files are missing.
"""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.uc21_rates import data as D
from src.uc21_rates import predict as P
from src.uc21_rates.baseline import LEVELS, MedianBaseline
from src.uc21_rates.metrics import coverage, mae, mape

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = D.RAW_CSV.exists()
HAVE_MODELS = (D.MODEL_DIR / "meta.json").exists() and all((D.MODEL_DIR / f).exists() for f in P.MODEL_FILES.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data/uc21_rates/raw/hotels.csv not downloaded")
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="models/uc21_rates not trained")

AGREED_FEATURES = {
    "hotel", "arrival_month", "arrival_week", "arrival_weekday", "lead_time", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "customer_type", "is_repeated_guest", "required_car_parking_spaces",
    "total_of_special_requests",
}

BOOKING = {"hotel": "City Hotel", "arrival_date": "2017-05-12", "nights": 3, "adults": 2, "children": 0,
           "room_type": "A", "meal": "BB", "channel": "TA/TO", "lead_time": 30}

FAKE_META = {
    "allowed": {"hotel": ["City Hotel", "Resort Hotel"], "room_type": ["A", "D"], "meal": ["BB", "HB", "SC"],
                "channel": ["Direct", "TA/TO"], "market_segment": ["Direct", "Online TA"],
                "customer_type": ["Transient", "Contract"]},
    "defaults": {"market_segment": "Online TA", "customer_type": "Transient", "is_repeated_guest": 0,
                 "parking": 0, "special_requests": 0, "babies": 0},
    "limits": {"nights": [1, 30], "adults": [0, 4], "children": [0, 3], "babies": [0, 2], "lead_time": [0, 700],
               "parking": [0, 2], "special_requests": [0, 5]},
}


def raw_frame(records):
    """A tiny table with the raw file's columns, for tests that must run without the data."""
    base = {"hotel": "City Hotel", "is_canceled": 0, "lead_time": 10, "arrival_date_year": 2016,
            "arrival_date_month": "May", "arrival_date_week_number": 20, "arrival_date_day_of_month": 10,
            "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2, "children": 0.0, "babies": 0,
            "meal": "BB", "country": "PRT", "market_segment": "Online TA", "distribution_channel": "TA/TO",
            "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A", "assigned_room_type": "A", "booking_changes": 0, "deposit_type": "No Deposit",
            "agent": 9.0, "company": None, "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 100.0,
            "required_car_parking_spaces": 0, "total_of_special_requests": 0, "reservation_status": "Check-Out",
            "reservation_status_date": "2016-05-13"}
    return pd.DataFrame([{**base, **r} for r in records])


# ---------- temporal leakage ----------

def test_features_are_exactly_the_agreed_booking_time_list():
    assert set(D.FEATURES) == AGREED_FEATURES
    assert len(D.FEATURES) == len(AGREED_FEATURES)


def test_leak_and_unused_columns_are_never_features():
    for col in D.LEAK_COLUMNS + D.NOT_USED_COLUMNS + [D.TARGET]:
        assert col not in D.FEATURES
    assert {"assigned_room_type", "reservation_status", "reservation_status_date", "booking_changes",
            "days_in_waiting_list", "is_canceled"} <= set(D.LEAK_COLUMNS)


def test_time_split_puts_2017_only_in_test():
    raw = raw_frame([{"arrival_date_year": y, "arrival_date_month": m} for y, m in
                     [(2015, "July"), (2016, "January"), (2016, "December"), (2017, "January"), (2017, "August")]])
    train, test = D.split(D.prepare(raw))
    assert set(train["arrival_date"].dt.year) == {2015, 2016}
    assert set(test["arrival_date"].dt.year) == {2017}
    assert not set(train["row_id"]) & set(test["row_id"])


def test_validation_slice_is_the_end_of_the_training_period():
    raw = raw_frame([{"arrival_date_year": 2016, "arrival_date_month": m} for m in ["March", "August", "September", "December"]])
    fit, valid = D.validation_split(D.prepare(raw))
    assert fit["arrival_date"].max() < pd.Timestamp(D.VALID_START) <= valid["arrival_date"].min()
    assert valid["arrival_date"].max() < pd.Timestamp("2017-01-01")


def test_outlier_limits_come_from_the_training_period_only():
    train_rows = [{"arrival_date_year": 2016, "adr": a} for a in (80.0, 90.0, 100.0)]
    calm = raw_frame(train_rows + [{"arrival_date_year": 2017, "adr": 95.0}])
    wild = raw_frame(train_rows + [{"arrival_date_year": 2017, "adr": 50000.0}])
    _, calm_report = D.clean(D.prepare(calm))
    wild_clean, wild_report = D.clean(D.prepare(wild))
    assert calm_report["outlier_rule"] == wild_report["outlier_rule"]
    assert wild_report["outlier_rule"]["training_median"] == 90.0
    assert wild_report["steps"][-1]["removed_test_period"] == 1
    assert 50000.0 not in set(wild_clean["adr"])


def test_cleaning_filters_remove_cancelled_free_and_zero_night_rows():
    raw = raw_frame([{}, {"is_canceled": 1}, {"adr": 0.0}, {"stays_in_weekend_nights": 0, "stays_in_week_nights": 0}])
    cleaned, report = D.clean(D.prepare(raw))
    assert len(cleaned) == 1
    assert [s["removed"] for s in report["steps"]] == [1, 1, 1, 0]


@needs_data
def test_real_training_rows_have_no_2017_arrivals():
    train, test, report = D.load_clean_split()
    assert train["arrival_date"].max() < pd.Timestamp("2017-01-01")
    assert test["arrival_date"].min() >= pd.Timestamp("2017-01-01")
    assert not set(train["row_id"]) & set(test["row_id"])
    assert sum(s["removed"] for s in report["steps"]) + report["rows_after_cleaning"] == report["raw_rows"]


@needs_data
@needs_models
def test_test_set_still_matches_its_lock():
    _, test, _ = D.load_clean_split()
    lock = json.loads((D.MODEL_DIR / "test_lock.json").read_text(encoding="utf-8"))
    assert lock["rows"] == len(test)
    assert lock["sha256"] == D.frame_fingerprint(test)


@needs_models
def test_saved_models_only_see_the_agreed_columns():
    models = P.load_models()
    for name in ("ridge", "gradient_boosting", "q10", "q90"):
        assert list(models[name].feature_names_in_) == D.FEATURES
    assert {c for level in LEVELS for c in level} <= set(D.FEATURES)


# ---------- metric correctness ----------

def test_mae_on_a_hand_computed_example():
    assert mae([100, 200, 300], [110, 190, 330]) == pytest.approx((10 + 10 + 30) / 3)


def test_mape_on_a_hand_computed_example():
    assert mape([100, 200], [110, 150]) == pytest.approx((10 + 25) / 2)


def test_mape_refuses_a_zero_actual():
    with pytest.raises(ValueError):
        mape([0, 100], [1, 100])


def test_coverage_counts_both_edges_as_inside():
    # 100 in [90, 100] yes (edge), 50 in [60, 70] no, 80 in [80, 95] yes (edge), 120 in [100, 110] no
    assert coverage([100, 50, 80, 120], [90, 60, 80, 100], [100, 70, 95, 110]) == pytest.approx(0.5)


def test_metrics_reject_bad_shapes():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        coverage([], [], [])


# ---------- baseline fallback ----------

def test_baseline_falls_back_level_by_level():
    X = pd.DataFrame({"hotel": ["City Hotel"] * 3 + ["Resort Hotel"] * 2,
                      "arrival_month": [5, 5, 6, 5, 5],
                      "reserved_room_type": ["A", "A", "D", "A", "A"]})
    y = [100.0, 120.0, 300.0, 50.0, 70.0]
    model = MedianBaseline().fit(X, y)
    query = pd.DataFrame({"hotel": ["City Hotel", "City Hotel", "City Hotel", "Other Hotel"],
                          "arrival_month": [5, 5, 12, 5],
                          "reserved_room_type": ["A", "G", "A", "A"]})
    pred, level = model.predict(query, return_level=True)
    assert level == ["hotel+month+room", "hotel+month", "hotel", "global"]
    assert pred.tolist() == [110.0, 110.0, 120.0, 100.0]


# ---------- derived nights, week number and input checks ----------

def test_weekend_nights_follow_the_rule_in_the_data_file():
    assert D.split_nights(date(2017, 1, 1), 1) == (1, 0)  # Sunday night
    assert D.split_nights(date(2017, 1, 2), 1) == (1, 0)  # Monday night
    assert D.split_nights(date(2016, 12, 31), 1) == (0, 1)  # Saturday night
    assert D.split_nights(date(2017, 5, 12), 3) == (1, 2)  # Fri, Sat, Sun
    assert D.split_nights(date(2017, 3, 8), 7) == (2, 5)


def test_week_number_matches_known_rows():
    assert D.week_number(date(2015, 7, 1)) == 27
    assert D.week_number(date(2015, 7, 5)) == 28  # Sunday starts a new week
    assert D.week_number(date(2016, 1, 3)) == 2
    assert D.week_number(date(2017, 1, 1)) == 1


@needs_data
def test_derived_nights_and_week_match_every_row_of_the_file():
    df = D.prepare(D.load_raw())
    combos = df[["arrival_date", "nights", "stays_in_weekend_nights", "stays_in_week_nights",
                 "arrival_week"]].drop_duplicates()
    for row in combos.itertuples(index=False):
        day = row.arrival_date.date()
        assert D.split_nights(day, row.nights) == (row.stays_in_weekend_nights, row.stays_in_week_nights)
        assert D.week_number(day) == row.arrival_week


def test_build_features_derives_dates_and_nights():
    row = P.build_features(BOOKING, FAKE_META).iloc[0]
    assert (row["arrival_month"], row["arrival_week"], row["arrival_weekday"]) == (5, 19, 4)
    assert (row["stays_in_weekend_nights"], row["stays_in_week_nights"]) == (1, 2)
    assert row["distribution_channel"] == "TA/TO" and row["reserved_room_type"] == "A"
    assert row["market_segment"] == "Online TA" and row["total_of_special_requests"] == 0
    assert list(P.build_features(BOOKING, FAKE_META).columns) == D.FEATURES


def test_build_features_accepts_numbers_typed_as_text():
    row = P.build_features({**BOOKING, "nights": "2", "adults": "1", "lead_time": 5.0}, FAKE_META).iloc[0]
    assert (row["adults"], row["lead_time"], row["stays_in_weekend_nights"] + row["stays_in_week_nights"]) == (1, 5, 2)


@pytest.mark.parametrize("change", [
    {"hotel": "Beach Hotel"}, {"room_type": "Z"}, {"meal": "Undefined"}, {"channel": "Phone"},
    {"arrival_date": "12/05/2017"}, {"arrival_date": "2017-02-30"}, {"arrival_date": "soon"},
    {"nights": 0}, {"nights": 31}, {"nights": 2.5}, {"adults": True}, {"adults": -1},
    {"adults": 0, "children": 0}, {"lead_time": "ten"}, {"is_repeated_guest": 2}, {"colour": "red"},
    {"market_segment": "Unknown"},
])
def test_build_features_rejects_bad_input(change):
    with pytest.raises(ValueError):
        P.build_features({**BOOKING, **change}, FAKE_META)


def test_build_features_rejects_missing_required_input():
    for key in P.REQUIRED:
        with pytest.raises(ValueError, match=key):
            P.build_features({k: v for k, v in BOOKING.items() if k != key}, FAKE_META)
    with pytest.raises(ValueError):
        P.build_features("not a dict", FAKE_META)


# ---------- predictions with the trained models ----------

@needs_models
def test_predict_rate_returns_a_rate_inside_its_range():
    out = P.predict_rate(BOOKING)
    assert set(out) == {"rate", "low", "high", "currency", "range_method"}
    assert out["currency"] == "EUR"
    assert out["low"] <= out["rate"] <= out["high"]
    assert 0 < out["rate"] < 1000


@needs_models
def test_predict_rate_raises_value_error_on_bad_input():
    with pytest.raises(ValueError):
        P.predict_rate({**BOOKING, "hotel": "Beach Hotel"})
    with pytest.raises(ValueError):
        P.predict_rate({**BOOKING, "arrival_date": "2017-13-01"})


@needs_models
def test_seasonality_returns_twelve_months():
    rows = P.seasonality(BOOKING)
    assert [r["month"] for r in rows] == list(range(1, 13))
    assert all(r["low"] <= r["rate"] <= r["high"] for r in rows)
    assert rows[4]["rate"] == P.predict_rate({**BOOKING, "arrival_date": "2017-05-15"})["rate"]


@needs_models
def test_allowed_values_and_defaults_come_from_meta():
    allowed = P.allowed_values()
    assert allowed["hotel"] == ["City Hotel", "Resort Hotel"]
    assert "Undefined" not in allowed["meal"] + allowed["channel"] + allowed["market_segment"]
    assert set(P.defaults()) == {"market_segment", "customer_type", "is_repeated_guest", "parking",
                                 "special_requests", "babies"}
    assert P.numeric_limits()["nights"][0] == 1


# ---------- demo page ----------

@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("uc21_rate_demo", ROOT / "app" / "uc21_rate_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_page_renders_and_never_inserts_html(demo):
    page = demo.render_page(demo.load_state()).decode("utf-8")
    for unsafe in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert unsafe not in page
    for placeholder in ("__STATE__", "__INFO__", "__INFO_CSS__", "__INFO_JS__", "__FEATURE_NAMES__",
                        "__MEAL_NAMES__", "__CHANNEL_NAMES__"):
        assert placeholder not in page
    for key in ("rate", "range", "season", "results", "drivers"):
        assert f'data-info-slot="{key}"' in page
    assert "createElementNS" in page and "textContent" in page


def test_demo_page_without_models_explains_what_to_do(demo, tmp_path):
    state = demo.load_state(tmp_path)
    assert state["ready"] is False and state["metrics"] is None
    assert "python -m src.uc21_rates.train" in demo.render_page(state).decode("utf-8")


def test_demo_json_cannot_close_the_script_block(demo):
    assert "</script>" not in demo._script_json({"x": "</script><b>"})


@needs_models
def test_demo_results_are_read_from_metrics_json(demo):
    metrics = json.loads((D.MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    state = demo.load_state()
    assert state["metrics"]["overall"] == metrics["overall"]
    assert state["metrics"]["top_drivers"] == metrics["top_drivers"][:8]
    assert json.dumps(metrics["overall"]["gradient_boosting"]["mae"]) in demo.render_page(state).decode("utf-8")


@needs_models
def test_demo_post_predict_works(demo):
    server = demo.make_server(0)  # any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body: bytes):
        req = urllib.request.Request(base + "/predict", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    try:
        with urllib.request.urlopen(base + "/", timeout=30) as res:
            assert res.status == 200 and "text/html" in res.headers["Content-Type"]
        status, out = post(json.dumps({**BOOKING, "nights": "3"}).encode("utf-8"))
        assert status == 200
        assert out["prediction"] == P.predict_rate(BOOKING)
        assert len(out["seasonality"]) == 12
        status, out = post(json.dumps({**BOOKING, "hotel": "<b>Beach</b>"}).encode("utf-8"))
        assert status == 400 and "hotel" in out["error"]
        status, out = post(b"not json")
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()


def test_why_this_price_swaps_each_input_for_its_typical_value():
    import pandas as pd

    from src.uc21_rates import explain as E
    from src.uc21_rates.data import MODEL_DIR

    train = pd.DataFrame({"hotel": ["City Hotel", "City Hotel", "Resort Hotel"], "arrival_month": [5, 5, 8],
                          "stays_in_weekend_nights": [1, 0, 2], "stays_in_week_nights": [2, 3, 5], "adults": [2, 2, 1],
                          "children": [0, 1, 0], "reserved_room_type": ["A", "A", "D"], "meal": ["BB", "HB", "BB"],
                          "distribution_channel": ["TA/TO"] * 3, "lead_time": [10, 30, 200],
                          "market_segment": ["Online TA"] * 3, "customer_type": ["Transient"] * 3,
                          "is_repeated_guest": [0, 0, 1], "required_car_parking_spaces": [0, 0, 1],
                          "total_of_special_requests": [0, 1, 2], "babies": [0, 0, 0]})
    typical = E.typical_values(train)
    assert typical["hotel"] == "City Hotel" and typical["arrival_month"] == 5 and typical["nights"] == 3
    assert typical["lead_time"] == 30 and typical["room_type"] == "A" and typical["meal"] == "BB"

    if not (MODEL_DIR / E.TYPICAL_FILE).exists() or not (MODEL_DIR / "meta.json").exists():
        pytest.skip("models or typical values not made on this machine")
    from src.uc21_rates.predict import predict_rate

    booking = {"hotel": "Resort Hotel", "arrival_date": "2017-08-12", "nights": 7, "adults": 2, "children": 1,
               "room_type": "D", "meal": "HB", "channel": "TA/TO", "lead_time": 120}
    rows = E.explain(booking)
    assert [abs(r["change"]) for r in rows] == sorted((abs(r["change"]) for r in rows), reverse=True)
    stored = E._typical(str(MODEL_DIR.resolve()))
    hotel = next(r for r in rows if r["input"] == "hotel")
    swapped = predict_rate({**booking, "hotel": stored["hotel"]})["rate"]
    assert hotel["change"] == pytest.approx(predict_rate(booking)["rate"] - swapped, abs=0.011)
    assert all(r["change"] == 0 for r in rows if r["value"] == r["typical"])
