"""Train the rate models and score the locked 2017 test set once.

    python -m src.uc21_rates.train                   full run: choices, lock, fit, score once, save
    python -m src.uc21_rates.train --validate-only   only the choices on the 2016 validation slice;
                                                     the 2017 test set is not touched

The full run refuses to score the test set again if models/uc21_rates/metrics.json already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from . import data as D
from .baseline import MedianBaseline
from .metrics import coverage, mae, mape
from .models import HGB_GRID, RIDGE_GRID, make_hgb, make_ridge

POINT_MODELS = ["baseline", "ridge", "gradient_boosting"]
LABELS = {"baseline": "Baseline (median)", "ridge": "Ridge", "gradient_boosting": "Gradient boosting"}
MODEL_FILES = {"baseline": "baseline.joblib", "ridge": "ridge.joblib", "gradient_boosting": "hgb.joblib",
               "q10": "hgb_q10.joblib", "q90": "hgb_q90.joblib"}
IMPORTANCE_SAMPLE = 5000
IMPORTANCE_REPEATS = 5


def choose_settings(train: pd.DataFrame) -> dict:
    """Pick the Ridge alpha and the boosting settings on the time-based validation slice."""
    fit, valid = D.validation_split(train)
    Xf, yf, Xv, yv = fit[D.FEATURES], fit[D.TARGET], valid[D.FEATURES], valid[D.TARGET]
    base_mae = mae(yv, MedianBaseline().fit(Xf, yf).predict(Xv))
    ridge_scores = {alpha: mae(yv, make_ridge(alpha).fit(Xf, yf).predict(Xv)) for alpha in RIDGE_GRID}
    best_alpha = min(ridge_scores, key=ridge_scores.get)
    hgb_scores = []
    for params in HGB_GRID:
        hgb_scores.append({"params": params, "mae": mae(yv, make_hgb(params).fit(Xf, yf).predict(Xv))})
    best = min(hgb_scores, key=lambda r: r["mae"])
    # The quantile models reuse the chosen settings; their validation coverage is reported, not tuned.
    q10 = make_hgb(best["params"], "quantile", 0.1).fit(Xf, yf).predict(Xv)
    q90 = make_hgb(best["params"], "quantile", 0.9).fit(Xf, yf).predict(Xv)
    return {
        "fit_rows": int(len(fit)), "valid_rows": int(len(valid)),
        "fit_period": [str(fit["arrival_date"].min().date()), str(fit["arrival_date"].max().date())],
        "valid_period": [str(valid["arrival_date"].min().date()), str(valid["arrival_date"].max().date())],
        "baseline_mae": base_mae,
        "ridge_mae_by_alpha": {str(a): s for a, s in ridge_scores.items()},
        "ridge_alpha": best_alpha,
        "hgb_results": hgb_scores,
        "hgb_params": best["params"],
        "hgb_mae": best["mae"],
        "range_coverage": coverage(yv, np.minimum(q10, q90), np.maximum(q10, q90)),
    }


def fit_final(train: pd.DataFrame, chosen: dict) -> dict:
    X, y = train[D.FEATURES], train[D.TARGET]
    return {
        "baseline": MedianBaseline().fit(X, y),
        "ridge": make_ridge(chosen["ridge_alpha"]).fit(X, y),
        "gradient_boosting": make_hgb(chosen["hgb_params"]).fit(X, y),
        "q10": make_hgb(chosen["hgb_params"], "quantile", 0.1).fit(X, y),
        "q90": make_hgb(chosen["hgb_params"], "quantile", 0.9).fit(X, y),
    }


def _scores(y, preds: dict, low, high, display_low, display_high) -> dict:
    out = {name: {"mae": mae(y, preds[name]), "mape": mape(y, preds[name])} for name in POINT_MODELS}
    out["range_10_90"] = {"coverage": coverage(y, low, high),
                          "display_coverage": coverage(y, display_low, display_high),
                          "mean_width": float(np.mean(high - low))}
    return out


def score_once(models: dict, test: pd.DataFrame) -> dict:
    X, y = test[D.FEATURES], test[D.TARGET].to_numpy()
    preds = {name: models[name].predict(X) for name in POINT_MODELS}
    q10, q90 = models["q10"].predict(X), models["q90"].predict(X)
    low, high = np.minimum(q10, q90), np.maximum(q10, q90)
    rate = preds["gradient_boosting"]
    # The demo shows the range widened so it always contains the point rate.
    display_low, display_high = np.minimum(low, rate), np.maximum(high, rate)
    result = {"overall": _scores(y, preds, low, high, display_low, display_high)}
    result["overall"]["range_10_90"]["crossed_quantiles"] = int(np.sum(q10 > q90))
    result["overall"]["range_10_90"]["rate_outside_range"] = int(np.sum((rate < low) | (rate > high)))
    result["per_hotel"] = {}
    hotels = test["hotel"].to_numpy()
    for hotel in sorted(set(hotels)):
        m = hotels == hotel
        part = _scores(y[m], {k: v[m] for k, v in preds.items()}, low[m], high[m], display_low[m], display_high[m])
        result["per_hotel"][hotel] = {"rows": int(m.sum()), **part}
    return result


def drivers(model, test: pd.DataFrame) -> tuple[list[dict], dict]:
    sample = test.sample(n=min(IMPORTANCE_SAMPLE, len(test)), random_state=D.SEED)
    r = permutation_importance(model, sample[D.FEATURES], sample[D.TARGET], scoring="neg_mean_absolute_error",
                               n_repeats=IMPORTANCE_REPEATS, random_state=D.SEED)
    rows = [{"feature": f, "mae_increase": float(m), "std": float(s)}
            for f, m, s in zip(D.FEATURES, r.importances_mean, r.importances_std)]
    rows.sort(key=lambda row: row["mae_increase"], reverse=True)
    setup = {"sample_rows": int(len(sample)), "n_repeats": IMPORTANCE_REPEATS, "random_state": D.SEED,
             "scoring": "neg_mean_absolute_error (shown as increase in MAE, EUR)"}
    return rows, setup


def _allowed(series: pd.Series) -> list[str]:
    return sorted(str(v) for v in series.unique() if v != "Undefined")


def build_meta(train: pd.DataFrame) -> dict:
    def mode(col):
        value = train[col].mode().iloc[0]
        return value.item() if hasattr(value, "item") else value

    def span(col):
        return [int(train[col].min()), int(train[col].max())]

    return {
        "features": D.FEATURES, "categorical": D.CATEGORICAL, "numeric": D.NUMERIC,
        "allowed": {
            "hotel": _allowed(train["hotel"]),
            "room_type": _allowed(train["reserved_room_type"]),
            "meal": _allowed(train["meal"]),
            "channel": _allowed(train["distribution_channel"]),
            "market_segment": _allowed(train["market_segment"]),
            "customer_type": _allowed(train["customer_type"]),
        },
        "defaults": {
            "market_segment": mode("market_segment"), "customer_type": mode("customer_type"),
            "is_repeated_guest": mode("is_repeated_guest"), "parking": mode("required_car_parking_spaces"),
            "special_requests": mode("total_of_special_requests"), "babies": mode("babies"),
        },
        "limits": {
            "nights": [1, int(train["nights"].max())], "adults": span("adults"), "children": span("children"),
            "babies": span("babies"), "lead_time": span("lead_time"),
            "parking": span("required_car_parking_spaces"), "special_requests": span("total_of_special_requests"),
        },
        "train_rows": int(len(train)),
        "train_period": [str(train["arrival_date"].min().date()), str(train["arrival_date"].max().date())],
        "currency": "EUR",
    }


def print_cleaning(report: dict) -> None:
    print(f"Raw rows: {report['raw_rows']}")
    print(f"{'filter':<22}{'removed':>9}{'train':>8}{'test':>8}{'left':>8}  reason")
    for s in report["steps"]:
        print(f"{s['filter']:<22}{s['removed']:>9}{s['removed_train_period']:>8}{s['removed_test_period']:>8}"
              f"{s['rows_left']:>8}  {s['reason']}")
    r = report["outlier_rule"]
    print(f"Outlier limits from the training period: median {r['training_median']:.2f} EUR, "
          f"keep {r['low']:.3f} to {r['high']:.2f} EUR")


def print_validation(chosen: dict) -> None:
    print(f"\nValidation slice (inside training): fit {chosen['fit_period'][0]}..{chosen['fit_period'][1]} "
          f"({chosen['fit_rows']} rows), check {chosen['valid_period'][0]}..{chosen['valid_period'][1]} "
          f"({chosen['valid_rows']} rows)")
    print(f"  baseline MAE {chosen['baseline_mae']:.2f}")
    for alpha, s in chosen["ridge_mae_by_alpha"].items():
        print(f"  ridge alpha={alpha:<6} MAE {s:.2f}")
    for r in chosen["hgb_results"]:
        p = r["params"]
        print(f"  boosting lr={p['learning_rate']} leaves={p['max_leaf_nodes']} iter={p['max_iter']}  MAE {r['mae']:.2f}")
    print(f"  chosen: ridge alpha={chosen['ridge_alpha']}, boosting {chosen['hgb_params']} "
          f"(validation MAE {chosen['hgb_mae']:.2f}); 10-90 range coverage on validation "
          f"{chosen['range_coverage'] * 100:.1f}%")


def print_results(metrics: dict) -> None:
    t = metrics["test_set"]
    print(f"\nLocked test set: {t['rows']} rows, arrivals {t['first_arrival']}..{t['last_arrival']}, sha256 {t['sha256']}")

    def table(title: str, block: dict) -> None:
        print(f"\n{title}")
        print(f"  {'model':<20}{'MAE (EUR)':>11}{'MAPE (%)':>10}")
        for name in POINT_MODELS:
            print(f"  {LABELS[name]:<20}{block[name]['mae']:>11.2f}{block[name]['mape']:>10.1f}")
        r = block["range_10_90"]
        print(f"  10-90% range coverage {r['coverage'] * 100:.1f}% (shown range {r['display_coverage'] * 100:.1f}%), "
              f"mean width {r['mean_width']:.2f} EUR")

    table("Test set, both hotels", metrics["overall"])
    for hotel, block in metrics["per_hotel"].items():
        table(f"Test set, {hotel} ({block['rows']} rows)", block)
    r = metrics["overall"]["range_10_90"]
    print(f"\n  crossed quantiles: {r['crossed_quantiles']}, point rate outside raw range: {r['rate_outside_range']}")
    print("\nTop 8 drivers of the gradient boosting model (permutation importance on "
          f"{metrics['importance_setup']['sample_rows']} test rows, increase in MAE, EUR):")
    for i, d in enumerate(metrics["top_drivers"][:8], 1):
        print(f"  {i}. {d['feature']:<30}{d['mae_increase']:>7.2f}  (+/- {d['std']:.2f})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the UC 2.1 hotel rate models.")
    parser.add_argument("--data", default=str(D.RAW_CSV))
    parser.add_argument("--model-dir", default=str(D.MODEL_DIR))
    parser.add_argument("--validate-only", action="store_true", help="stop before the test set is touched")
    parser.add_argument("--rescore", action="store_true",
                        help="allow scoring the locked test set again (not for normal use)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = time.time()
    model_dir = Path(args.model_dir)
    lock_path, metrics_path = model_dir / "test_lock.json", model_dir / "metrics.json"
    if not args.validate_only and metrics_path.exists() and not args.rescore:
        print(f"{metrics_path} exists: the locked test set was already scored once. Not scoring it again.")
        return 2

    cleaned, report = D.clean(D.prepare(D.load_raw(args.data)))
    train, test = D.split(cleaned)
    print_cleaning(report)
    print(f"Train: {len(train)} rows, arrivals {train['arrival_date'].min().date()}..{train['arrival_date'].max().date()}")
    print(f"Test:  {len(test)} rows (arrivals in {D.TEST_YEAR})")
    chosen = choose_settings(train)
    print_validation(chosen)
    if args.validate_only:
        print(f"\nValidate-only run: the test set was not used. ({time.time() - started:.0f} s)")
        return 0

    # Lock the test set before any model sees it.
    fingerprint = D.frame_fingerprint(test)
    model_dir.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["sha256"] != fingerprint or lock["rows"] != len(test):
            print("The test set does not match test_lock.json. Stopping.")
            return 2
    else:
        lock = {"rows": int(len(test)), "sha256": fingerprint, "locked_on": datetime.now().isoformat(timespec="seconds"),
                "test_period": f"arrivals {test['arrival_date'].min().date()}..{test['arrival_date'].max().date()}",
                "how": "SHA-256 of the cleaned test rows, sorted by row_id (row number in the raw file, from 0), "
                       "columns in this order, written with pandas to_csv(index=False, lineterminator='\\n', "
                       "float_format='%.6f'), dates as YYYY-MM-DD",
                "columns": D.FINGERPRINT_COLUMNS, "data_file_sha256": "see data/uc21_rates/raw/SHA256SUMS.txt"}
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    models = fit_final(train, chosen)
    result = score_once(models, test)
    top, setup = drivers(models["gradient_boosting"], test)
    meta = build_meta(train)
    for name, file in MODEL_FILES.items():
        joblib.dump(models[name], model_dir / file, compress=3)
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    metrics = {
        "test_set": {"rows": int(len(test)), "sha256": fingerprint,
                     "first_arrival": str(test["arrival_date"].min().date()),
                     "last_arrival": str(test["arrival_date"].max().date())},
        "train_set": {"rows": int(len(train)), "first_arrival": meta["train_period"][0],
                      "last_arrival": meta["train_period"][1]},
        **result,
        "top_drivers": top,
        "importance_setup": setup,
        "validation": chosen,
        "cleaning": report,
        "scored_on": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_results(metrics)
    print(f"\nSaved models, meta.json, metrics.json and test_lock.json in {model_dir} ({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
