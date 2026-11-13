"""Tests for the conformal calibration of the 10-90% range."""

import json

import numpy as np
import pytest

from src.uc21_rates import calibrate as C


def test_conformity_score_is_the_distance_outside_the_range():
    scores = C.conformity_scores([100, 50, 200], [80, 60, 90], [120, 70, 180])
    assert list(scores) == [-20, 10, 20]  # inside by 20, below by 10, above by 20


def test_conformal_quantile_by_hand():
    scores = np.arange(1, 10)  # 1..9, n = 9: k = ceil(10 * 0.8) = 8 -> the 8th smallest
    assert C.conformal_quantile(scores, 0.2) == 8
    assert C.conformal_quantile([5, 1, 3], 0.2) == float("inf")  # k = ceil(4 * 0.8) = 4 > 3


def test_calibrated_range_widens_or_narrows_both_ends():
    low, high = C.calibrated_range([80, 90], [120, 100], 5)
    assert list(low) == [75, 85] and list(high) == [125, 105]
    low, high = C.calibrated_range([80], [120], -50)  # a negative Q that would cross is kept ordered
    assert low[0] <= high[0]


def test_calibrated_range_reaches_its_target_on_exchangeable_data():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(100, 20, 2000), rng.normal(100, 20, 2000)
    q10, q90 = np.full(2000, 95.0), np.full(2000, 105.0)  # a far too narrow raw range
    q = C.conformal_quantile(C.conformity_scores(y_cal, q10, q90))
    low, high = C.calibrated_range(q10, q90, q)
    held = np.mean((y_new >= low) & (y_new <= high))
    assert 0.77 < held < 0.83


def test_scoring_refuses_a_second_time(tmp_path, capsys):
    (tmp_path / C.SCORED_FILE).write_text("{}", encoding="utf-8")
    assert C.main(["--model-dir", str(tmp_path)]) == 2
    assert "already scored once" in capsys.readouterr().out


def test_predictions_use_the_calibrated_range_when_present():
    from src.uc21_rates import data as D
    from src.uc21_rates import predict as P

    if not (D.MODEL_DIR / C.CALIBRATION_FILE).exists():
        pytest.skip("calibration not run on this machine")
    q = json.loads((D.MODEL_DIR / C.CALIBRATION_FILE).read_text(encoding="utf-8"))["q"]
    booking = {"hotel": "City Hotel", "arrival_date": "2017-05-12", "nights": 3, "adults": 2, "children": 0,
               "room_type": "A", "meal": "BB", "channel": "TA/TO", "lead_time": 30}
    result = P.predict_rate(booking)
    assert result["range_method"] == "calibrated" and result["low"] <= result["rate"] <= result["high"]
    models = P.load_models()
    X = P.build_features(booking, models["meta"])
    assert result["high"] >= round(float(models["q90_cal"].predict(X)[0]) + q, 2) - 0.01
