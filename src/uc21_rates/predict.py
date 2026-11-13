"""Predict a reference nightly rate (EUR) and a 10-90% range for one booking.

    from src.uc21_rates.predict import predict_rate, seasonality, allowed_values
    predict_rate({"hotel": "City Hotel", "arrival_date": "2017-05-12", "nights": 3, "adults": 2,
                  "children": 0, "room_type": "A", "meal": "BB", "channel": "TA/TO", "lead_time": 30})

Required keys: hotel, arrival_date (YYYY-MM-DD), nights, adults, children, room_type, meal,
channel, lead_time. Optional keys: market_segment, customer_type, is_repeated_guest, parking,
special_requests, babies. A missing optional key takes the most common value in the training
data (see defaults(); for the current models: market_segment "Online TA", customer_type
"Transient", is_repeated_guest 0, parking 0, special_requests 0, babies 0).

Allowed values and number ranges come from the training data (meta.json, written by train.py),
so a UI or an API can validate input with allowed_values() and numeric_limits().
When calibrate.py has been run, the range is the conformally calibrated one (range_method
"calibrated"); otherwise it comes straight from the two quantile models ("raw").
The model has no year input: rates are at 2015-2016 price levels, for two Portuguese hotels.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES, MODEL_DIR, split_nights, week_number

REQUIRED = ["hotel", "arrival_date", "nights", "adults", "children", "room_type", "meal", "channel", "lead_time"]
OPTIONAL = ["market_segment", "customer_type", "is_repeated_guest", "parking", "special_requests", "babies"]
CATEGORICAL_INPUTS = ["hotel", "room_type", "meal", "channel", "market_segment", "customer_type"]
COUNT_INPUTS = ["nights", "adults", "children", "babies", "lead_time", "parking", "special_requests"]
MODEL_FILES = {"baseline": "baseline.joblib", "ridge": "ridge.joblib", "gradient_boosting": "hgb.joblib",
               "q10": "hgb_q10.joblib", "q90": "hgb_q90.joblib"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve(model_dir) -> str:
    return str(Path(model_dir).resolve())


@lru_cache(maxsize=4)
def _load_meta(model_dir: str) -> dict:
    path = Path(model_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.uc21_rates.train")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_models(model_dir: str) -> dict:
    models = {}
    for name, file in MODEL_FILES.items():
        path = Path(model_dir) / file
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train first: python -m src.uc21_rates.train")
        models[name] = joblib.load(path)
    models["meta"] = _load_meta(model_dir)
    calibration = Path(model_dir) / "calibration.json"
    cal_files = {"q10_cal": "hgb_q10_cal.joblib", "q90_cal": "hgb_q90_cal.joblib"}
    if calibration.exists() and all((Path(model_dir) / f).exists() for f in cal_files.values()):
        for name, file in cal_files.items():
            models[name] = joblib.load(Path(model_dir) / file)
        models["cal_q"] = float(json.loads(calibration.read_text(encoding="utf-8"))["q"])
    return models


def load_models(model_dir=MODEL_DIR) -> dict:
    """Load the saved models once (cached): baseline, ridge, gradient_boosting, q10, q90, meta."""
    return _load_models(_resolve(model_dir))


def allowed_values(model_dir=MODEL_DIR) -> dict:
    """Allowed values for each text input, as seen in the training data."""
    return {k: list(v) for k, v in _load_meta(_resolve(model_dir))["allowed"].items()}


def defaults(model_dir=MODEL_DIR) -> dict:
    """Values used for missing optional inputs (most common in the training data)."""
    return dict(_load_meta(_resolve(model_dir))["defaults"])


def numeric_limits(model_dir=MODEL_DIR) -> dict:
    """[lowest, highest] value seen in the training data for each number input."""
    return {k: list(v) for k, v in _load_meta(_resolve(model_dir))["limits"].items()}


def _count(name: str, value, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number, not true/false")
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        value = int(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a whole number")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high} (range seen in the training data)")
    return value


def _arrival(value) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not DATE_PATTERN.match(value.strip()):
        raise ValueError("arrival_date must be a date written as YYYY-MM-DD")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError("arrival_date is not a real calendar date") from None


def build_features(booking: dict, meta: dict) -> pd.DataFrame:
    """Check one booking and turn it into the one-row feature table the models expect."""
    if not isinstance(booking, dict):
        raise ValueError("booking must be a dict")
    unknown = sorted(set(booking) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise ValueError(f"unknown input(s): {', '.join(unknown)}")
    missing = [k for k in REQUIRED if booking.get(k) is None or booking.get(k) == ""]
    if missing:
        raise ValueError(f"missing input(s): {', '.join(missing)}")
    b = {**meta["defaults"], **{k: v for k, v in booking.items() if v is not None and v != ""}}
    allowed, limits = meta["allowed"], meta["limits"]
    for key in CATEGORICAL_INPUTS:
        if not isinstance(b[key], str) or b[key] not in allowed[key]:
            raise ValueError(f"{key} must be one of: {', '.join(allowed[key])}")
    counts = {key: _count(key, b[key], *limits[key]) for key in COUNT_INPUTS}
    if counts["adults"] + counts["children"] < 1:
        raise ValueError("a booking needs at least one adult or child")
    repeated = b["is_repeated_guest"]
    if repeated not in (0, 1, True, False) or isinstance(repeated, float) and not repeated.is_integer():
        raise ValueError("is_repeated_guest must be 0 or 1")
    arrival = _arrival(b["arrival_date"])
    weekend, week = split_nights(arrival, counts["nights"])
    row = {
        "hotel": b["hotel"], "meal": b["meal"], "market_segment": b["market_segment"],
        "distribution_channel": b["channel"], "reserved_room_type": b["room_type"],
        "customer_type": b["customer_type"],
        "arrival_month": arrival.month, "arrival_week": week_number(arrival), "arrival_weekday": arrival.weekday(),
        "lead_time": counts["lead_time"], "stays_in_weekend_nights": weekend, "stays_in_week_nights": week,
        "adults": counts["adults"], "children": counts["children"], "babies": counts["babies"],
        "is_repeated_guest": int(repeated), "required_car_parking_spaces": counts["parking"],
        "total_of_special_requests": counts["special_requests"],
    }
    return pd.DataFrame([row], columns=FEATURES)


def _rates(models: dict, X: pd.DataFrame) -> list[dict]:
    rate = models["gradient_boosting"].predict(X)
    if "cal_q" in models:  # conformalized quantile regression, see calibrate.py
        q10, q90 = models["q10_cal"].predict(X) - models["cal_q"], models["q90_cal"].predict(X) + models["cal_q"]
        method = "calibrated"
    else:
        q10, q90 = models["q10"].predict(X), models["q90"].predict(X)
        method = "raw"
    out = []
    for r, a, b in zip(rate, q10, q90):
        low, high = min(a, b), max(a, b)
        # low <= rate <= high is enforced only by clipping the displayed range to include the rate.
        # The rate itself is never changed, and the range is not a calibrated interval.
        low, high = min(low, r), max(high, r)
        out.append({"rate": round(float(r), 2), "low": round(float(low), 2), "high": round(float(high), 2),
                    "currency": "EUR", "range_method": method})
    return out


def predict_rate(booking: dict, model_dir=MODEL_DIR) -> dict:
    """Return {"rate", "low", "high", "currency": "EUR"} for one booking. Raises ValueError on bad input."""
    models = load_models(model_dir)
    return _rates(models, build_features(booking, models["meta"]))[0]


def seasonality(booking: dict, model_dir=MODEL_DIR) -> list[dict]:
    """The same booking moved to the 15th of each month of 2017: 12 x {month, rate, low, high}."""
    models = load_models(model_dir)
    frames = [build_features({**booking, "arrival_date": f"2017-{m:02d}-15"}, models["meta"]) for m in range(1, 13)]
    rates = _rates(models, pd.concat(frames, ignore_index=True))
    return [{"month": m, "rate": r["rate"], "low": r["low"], "high": r["high"]} for m, r in zip(range(1, 13), rates)]
