"""Calibrate the 10-90% range with conformalized quantile regression (CQR).

Everything below was fixed before the 2017 test set was used for this step:
1. Calibration slice: arrivals 2016-09-01 to 2016-12-31 (the last part of the training period).
2. The 10% and 90% quantile models are refitted on the rest of 2015-2016 (arrivals before
   2016-09-01), with the settings already chosen for the final models (metrics.json).
3. Conformity score of each calibration booking: E = max(q10 - actual, actual - q90).
4. Q = the ceil((n + 1) * 0.8)-th smallest score (target coverage 80%, n = calibration rows).
5. Calibrated range for a new booking: [q10 - Q, q90 + Q].

The locked 2017 test set may be scored exactly once for the calibrated range: the run checks the
test fingerprint against test_lock.json and refuses if calibration_metrics.json already exists.
Point-rate results (metrics.json) are not touched.

The guarantee of CQR only holds when new bookings behave like the calibration bookings. 2017
prices were higher than 2016 prices, so the coverage on 2017 can still fall short of 80%.

Usage:
    python -m src.uc21_rates.calibrate              fit, then score the 2017 set once
    python -m src.uc21_rates.calibrate --fit-only   fit and save; the 2017 set is not touched
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data as D
from .metrics import coverage
from .models import make_hgb

ALPHA = 0.2  # 1 - target coverage
CAL_FILES = {"q10_cal": "hgb_q10_cal.joblib", "q90_cal": "hgb_q90_cal.joblib"}
CALIBRATION_FILE = "calibration.json"
SCORED_FILE = "calibration_metrics.json"


def conformity_scores(y, q10, q90) -> np.ndarray:
    """How far each actual rate lies outside its raw range (negative when it is inside)."""
    y, q10, q90 = (np.asarray(v, dtype=float) for v in (y, q10, q90))
    return np.maximum(q10 - y, y - q90)


def conformal_quantile(scores, alpha: float = ALPHA) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; infinite if n is too small."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(scores[k - 1])


def calibrated_range(q10, q90, q: float) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.asarray(q10, dtype=float) - q, np.asarray(q90, dtype=float) + q
    return np.minimum(low, high), np.maximum(low, high)


def fit(train: pd.DataFrame, params: dict, alpha: float = ALPHA) -> tuple[dict, dict]:
    """Refit the quantile models without the calibration slice and work out Q on it."""
    fit_part, cal = D.validation_split(train)
    X, y = fit_part[D.FEATURES], fit_part[D.TARGET]
    models = {"q10_cal": make_hgb(params, "quantile", 0.1).fit(X, y),
              "q90_cal": make_hgb(params, "quantile", 0.9).fit(X, y)}
    q10, q90 = models["q10_cal"].predict(cal[D.FEATURES]), models["q90_cal"].predict(cal[D.FEATURES])
    y_cal = cal[D.TARGET].to_numpy()
    q = conformal_quantile(conformity_scores(y_cal, q10, q90), alpha)
    low, high = calibrated_range(q10, q90, q)
    info = {
        "method": "conformalized quantile regression (CQR)",
        "alpha": alpha, "target_coverage": 1 - alpha, "q": q,
        "fit_rows": int(len(fit_part)),
        "fit_period": [str(fit_part["arrival_date"].min().date()), str(fit_part["arrival_date"].max().date())],
        "calibration_rows": int(len(cal)),
        "calibration_period": [str(cal["arrival_date"].min().date()), str(cal["arrival_date"].max().date())],
        "settings": params,
        "calibration_slice_raw_coverage": coverage(y_cal, np.minimum(q10, q90), np.maximum(q10, q90)),
        "calibration_slice_calibrated_coverage": coverage(y_cal, low, high),
        "fitted_on": datetime.now().isoformat(timespec="seconds"),
    }
    return models, info


def score_once(models: dict, q: float, rate_model, test: pd.DataFrame) -> dict:
    """Coverage and width of the calibrated range on the locked test set, overall and per hotel."""
    X, y = test[D.FEATURES], test[D.TARGET].to_numpy()
    low, high = calibrated_range(models["q10_cal"].predict(X), models["q90_cal"].predict(X), q)
    rate = rate_model.predict(X)
    show_low, show_high = np.minimum(low, rate), np.maximum(high, rate)  # the demo always shows the rate inside

    def part(mask) -> dict:
        return {"rows": int(mask.sum()), "coverage": coverage(y[mask], low[mask], high[mask]),
                "display_coverage": coverage(y[mask], show_low[mask], show_high[mask]),
                "mean_width": float(np.mean(high[mask] - low[mask]))}

    hotels = test["hotel"].to_numpy()
    out = {"overall": part(np.ones(len(y), dtype=bool)),
           "per_hotel": {h: part(hotels == h) for h in sorted(set(hotels))}}
    out["overall"]["rate_outside_range"] = int(np.sum((rate < low) | (rate > high)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(D.RAW_CSV))
    parser.add_argument("--model-dir", default=str(D.MODEL_DIR))
    parser.add_argument("--fit-only", action="store_true", help="do not touch the 2017 test set")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model_dir = Path(args.model_dir)
    scored_path = model_dir / SCORED_FILE
    if not args.fit_only and scored_path.exists():
        print(f"{scored_path} exists: the calibrated range was already scored once. Not scoring it again.")
        return 2
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    params = metrics["validation"]["hgb_params"]

    cleaned, _ = D.clean(D.prepare(D.load_raw(args.data)))
    train, test = D.split(cleaned)
    models, info = fit(train, params)
    for name, file in CAL_FILES.items():
        joblib.dump(models[name], model_dir / file, compress=3)
    (model_dir / CALIBRATION_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Calibration slice {info['calibration_period'][0]}..{info['calibration_period'][1]} "
          f"({info['calibration_rows']} rows); quantile models refitted on {info['fit_rows']} rows.")
    print(f"Q = {info['q']:.2f} EUR. On the calibration slice: raw range held "
          f"{info['calibration_slice_raw_coverage']:.1%}, calibrated range {info['calibration_slice_calibrated_coverage']:.1%}.")
    if args.fit_only:
        print("Fit-only run: the 2017 test set was not used.")
        return 0

    lock = json.loads((model_dir / "test_lock.json").read_text(encoding="utf-8"))
    if D.frame_fingerprint(test) != lock["sha256"] or len(test) != lock["rows"]:
        print("The test set does not match test_lock.json. Stopping.")
        return 2
    result = score_once(models, info["q"], joblib.load(model_dir / "hgb.joblib"), test)
    scored = {"method": info["method"], "q": info["q"], "target_coverage": info["target_coverage"],
              "test_set": {"rows": lock["rows"], "sha256": lock["sha256"]},
              **result, "scored_on": datetime.now().isoformat(timespec="seconds")}
    scored_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    o = result["overall"]
    print(f"Locked 2017 test set, scored once: calibrated range held {o['coverage']:.1%} "
          f"(mean width {o['mean_width']:.2f} EUR).")
    for hotel, p in result["per_hotel"].items():
        print(f"  {hotel}: {p['coverage']:.1%}, mean width {p['mean_width']:.2f} EUR ({p['rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
