"""Load, clean and split the hotel booking data for the rate model.

Everything here is plain pandas. The cleaning steps keep a count of the rows each one removes,
and the outlier limits are worked out from the training period only.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "uc21_rates" / "raw" / "hotels.csv"
MODEL_DIR = ROOT / "models" / "uc21_rates"

SEED = 0
TARGET = "adr"
TRAIN_YEARS = (2015, 2016)
TEST_YEAR = 2017
VALID_START = "2016-09-01"  # hyperparameter choices use 2016-09-01..2016-12-31 only
OUTLIER_FACTOR = 10  # an adr below median/10 or above median*10 (training median) is treated as an error

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MONTH_NUMBER = {name: i for i, name in enumerate(MONTHS, start=1)}

# Only information known when the booking is made.
CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type", "customer_type"]
NUMERIC = ["arrival_month", "arrival_week", "arrival_weekday", "lead_time",
           "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
           "is_repeated_guest", "required_car_parking_spaces", "total_of_special_requests"]
FEATURES = CATEGORICAL + NUMERIC

# Known only after the booking was made, or the outcome itself: never used as features.
LEAK_COLUMNS = ["assigned_room_type", "reservation_status", "reservation_status_date",
                "booking_changes", "days_in_waiting_list", "is_canceled"]
# Not in the agreed feature list for this first version.
NOT_USED_COLUMNS = ["country", "agent", "company", "deposit_type",
                    "previous_cancellations", "previous_bookings_not_canceled"]

FINGERPRINT_COLUMNS = ["row_id", "arrival_date"] + FEATURES + [TARGET]

# In this data file stays_in_weekend_nights counts the nights that START on a Sunday or a Monday.
# This matches every row of the file (checked in the tests). The paper calls them Saturday or
# Sunday nights; we follow the file so that predictions get the same kind of input as training.
WEEKEND_NIGHT_WEEKDAYS = (6, 0)  # Python weekday(): Monday = 0, Sunday = 6


def split_nights(arrival: date, nights: int) -> tuple[int, int]:
    """Return (weekend nights, week nights) for a stay, using the rule found in the data file."""
    weekend = sum(1 for k in range(nights) if (arrival + timedelta(days=k)).weekday() in WEEKEND_NIGHT_WEEKDAYS)
    return weekend, nights - weekend


def week_number(day: date) -> int:
    """Arrival week number as in the data file: weeks start on Sunday, week 1 contains 1 January."""
    jan1 = date(day.year, 1, 1)
    offset = (jan1.weekday() + 1) % 7  # weekday of 1 January counted from Sunday = 0
    return (day.timetuple().tm_yday - 1 + offset) // 7 + 1


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. See docs/uc21_results.md for the download.")
    return pd.read_csv(path)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the date columns and tidy two inputs. Removes no rows."""
    df = raw.copy()
    df["row_id"] = np.arange(len(df))
    df.index = df["row_id"].to_numpy()
    month = df["arrival_date_month"].map(MONTH_NUMBER)
    if month.isna().any():
        raise ValueError("Unknown month name in arrival_date_month")
    df["arrival_date"] = pd.to_datetime(pd.DataFrame({"year": df["arrival_date_year"], "month": month,
                                                      "day": df["arrival_date_day_of_month"]}))
    df["arrival_month"] = month.astype(int)
    df["arrival_week"] = df["arrival_date_week_number"].astype(int)
    df["arrival_weekday"] = df["arrival_date"].dt.day_of_week.astype(int)
    df["children"] = df["children"].fillna(0).astype(int)  # 4 missing values in the raw file
    # The paper defines meal "Undefined/SC" as one category (no meal package).
    df["meal"] = df["meal"].replace({"Undefined": "SC"})
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df


def _period_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    year = df["arrival_date"].dt.year
    return year.isin(TRAIN_YEARS), year == TEST_YEAR


def outlier_limits(train_rows: pd.DataFrame) -> dict:
    """Limits for the outlier rule, worked out from training-period rows only."""
    median = float(train_rows[TARGET].median())
    return {"training_median": median, "factor": OUTLIER_FACTOR,
            "low": median / OUTLIER_FACTOR, "high": median * OUTLIER_FACTOR}


def clean(prepared: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the filters in order and count what each one removes (overall, train and test period)."""
    steps = []
    df = prepared

    def keep(frame: pd.DataFrame, mask: pd.Series, name: str, reason: str) -> pd.DataFrame:
        train_mask, test_mask = _period_masks(frame)
        drop = ~mask
        steps.append({"filter": name, "reason": reason, "removed": int(drop.sum()),
                      "removed_train_period": int((drop & train_mask).sum()),
                      "removed_test_period": int((drop & test_mask).sum()),
                      "rows_left": int(mask.sum())})
        return frame[mask]

    df = keep(df, df["is_canceled"] == 0, "cancelled bookings",
              "the stay never happened, so its rate is not a realised nightly rate")
    df = keep(df, df[TARGET] > 0, "adr of 0 or less",
              "free or corrected stays, not a paid nightly rate")
    df = keep(df, df["nights"] >= 1, "zero nights",
              "no night was booked, so there is no nightly rate")
    train_mask, _ = _period_masks(df)
    limits = outlier_limits(df[train_mask])
    df = keep(df, df[TARGET].between(limits["low"], limits["high"]), "outlier adr",
              f"adr below training median / {OUTLIER_FACTOR} or above training median x {OUTLIER_FACTOR}")
    report = {"raw_rows": int(len(prepared)), "steps": steps, "outlier_rule": limits,
              "rows_after_cleaning": int(len(df))}
    return df, report


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time split: train = arrivals in 2015 and 2016, test = arrivals in 2017. No random split."""
    train_mask, test_mask = _period_masks(df)
    return df[train_mask], df[test_mask]


def validation_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inside the training period: fit on arrivals before VALID_START, check on the rest of 2016."""
    start = pd.Timestamp(VALID_START)
    return train[train["arrival_date"] < start], train[train["arrival_date"] >= start]


def frame_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the rows written as CSV in a fixed column and row order."""
    out = df.sort_values("row_id")[FINGERPRINT_COLUMNS].copy()
    out["arrival_date"] = out["arrival_date"].dt.strftime("%Y-%m-%d")
    text = out.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_clean_split(path: Path | str = RAW_CSV):
    """Convenience: raw file -> (train, test, cleaning report)."""
    cleaned, report = clean(prepare(load_raw(path)))
    train, test = split(cleaned)
    return train, test, report
