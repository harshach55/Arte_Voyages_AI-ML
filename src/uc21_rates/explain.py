"""Why this price? A simple what-if for one prediction.

For each input of a booking, the input is swapped for its typical training value (the median for
numbers, the most common value otherwise), the same model predicts again, and the difference is
reported: change = this prediction - prediction with the typical value. A positive change means
this input pushes the price up compared with a typical booking.

This is a what-if on the existing model, not a guarantee and not a cause: inputs interact, so the
changes do not add up to the price.

The typical values come from the training rows only (arrivals 2015-2016), never from the 2017 test
set. They are saved once in models/uc21_rates/typical.json:
    python -m src.uc21_rates.explain --save
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import data as D
from .predict import OPTIONAL, REQUIRED, defaults, predict_rate

TYPICAL_FILE = "typical.json"
LABELS = {"hotel": "Hotel", "arrival_date": "Arrival month", "nights": "Nights", "adults": "Adults",
          "children": "Children", "room_type": "Room type", "meal": "Meal", "channel": "Channel",
          "lead_time": "Lead time (days)", "market_segment": "Market segment", "customer_type": "Customer type",
          "is_repeated_guest": "Repeat guest", "parking": "Parking spaces", "special_requests": "Special requests",
          "babies": "Babies"}


def typical_values(train: pd.DataFrame) -> dict:
    """Median of each number input and most common value of each other input, from training rows."""
    def mode(column):
        return train[column].mode().iloc[0]

    def median(series):
        return int(round(float(series.median())))

    nights = train["stays_in_weekend_nights"] + train["stays_in_week_nights"]
    return {"hotel": str(mode("hotel")), "arrival_month": int(mode("arrival_month")),
            "nights": median(nights), "adults": median(train["adults"]), "children": median(train["children"]),
            "room_type": str(mode("reserved_room_type")), "meal": str(mode("meal")),
            "channel": str(mode("distribution_channel")), "lead_time": median(train["lead_time"]),
            "market_segment": str(mode("market_segment")), "customer_type": str(mode("customer_type")),
            "is_repeated_guest": int(mode("is_repeated_guest")), "parking": median(train["required_car_parking_spaces"]),
            "special_requests": median(train["total_of_special_requests"]), "babies": median(train["babies"])}


@lru_cache(maxsize=4)
def _typical(model_dir: str) -> dict:
    return json.loads((Path(model_dir) / TYPICAL_FILE).read_text(encoding="utf-8"))


def _swap(booking: dict, key: str, typical: dict) -> tuple[dict, str]:
    """The booking with one input set to its typical value, and that value as text."""
    if key == "arrival_date":  # the same date moved to the 15th of the most common arrival month
        year = date.fromisoformat(str(booking["arrival_date"]).strip()).year
        month = typical["arrival_month"]
        return {**booking, "arrival_date": f"{year}-{month:02d}-15"}, date(2000, month, 1).strftime("%B")
    return {**booking, key: typical[key]}, str(typical[key])


def explain(booking: dict, model_dir=D.MODEL_DIR) -> list[dict]:
    """One row per input: its value, the typical value and the change in EUR, largest change first.

    Returns an empty list when typical.json has not been made. Raises ValueError on a bad booking.
    """
    if not (Path(model_dir) / TYPICAL_FILE).exists():
        return []
    typical = _typical(str(Path(model_dir).resolve()))
    full = {**defaults(model_dir), **{k: v for k, v in booking.items() if v is not None and v != ""}}
    rate = predict_rate(full, model_dir)["rate"]
    rows = []
    for key in REQUIRED + OPTIONAL:
        swapped, shown = _swap(full, key, typical)
        try:
            other = predict_rate(swapped, model_dir)["rate"]
        except ValueError:  # e.g. no adults and no children after the swap
            continue
        value = full[key]
        if key == "arrival_date":
            value = date.fromisoformat(str(value).strip()).strftime("%B")
        rows.append({"input": key, "label": LABELS[key], "value": str(value), "typical": shown,
                     "change": round(rate - other, 2)})
    return sorted(rows, key=lambda r: -abs(r["change"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", action="store_true", help="work out the typical values and save them")
    parser.add_argument("--model-dir", default=str(D.MODEL_DIR))
    args = parser.parse_args(argv)
    if not args.save:
        parser.print_help()
        return 0
    train, _, _ = D.load_clean_split()  # only the training rows are used below
    typical = typical_values(train)
    typical["from"] = f"training rows only: {len(train)} bookings, arrivals 2015-2016"
    (Path(args.model_dir) / TYPICAL_FILE).write_text(json.dumps(typical, indent=2), encoding="utf-8")
    print(json.dumps(typical, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
