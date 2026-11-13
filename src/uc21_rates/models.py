"""The two learned models: Ridge and histogram gradient boosting (point and quantile)."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, SEED

# Ridge is linear, so month and weekday are one-hot encoded; a straight line cannot follow seasons.
RIDGE_ONEHOT = CATEGORICAL + ["arrival_month", "arrival_weekday"]
RIDGE_SCALED = [c for c in NUMERIC if c not in RIDGE_ONEHOT]

RIDGE_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0]
HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": iters}
    for lr in (0.05, 0.1) for leaves in (7, 15, 31, 63) for iters in (100, 200, 500)
]


def make_ridge(alpha: float = 1.0) -> Pipeline:
    prep = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), RIDGE_ONEHOT),
        ("scale", StandardScaler(), RIDGE_SCALED),
    ])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=alpha))])


def make_hgb(params: dict, loss: str = "squared_error", quantile: float | None = None) -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    extra = {"quantile": quantile} if loss == "quantile" else {}
    model = HistGradientBoostingRegressor(
        loss=loss, categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=False,  # no hidden random validation split; choices use the time slice
        random_state=SEED, **params, **extra)
    return Pipeline([("prep", prep), ("model", model)])
