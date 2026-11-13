"""Error measures for the rate model: MAE, MAPE and coverage of a price range."""

from __future__ import annotations

import numpy as np


def _arrays(*values):
    arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    if any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("inputs must have the same length")
    if len(arrays[0]) == 0:
        raise ValueError("inputs must not be empty")
    return arrays


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the unit of the target (EUR)."""
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent. Every actual value must be non-zero."""
    y, p = _arrays(y_true, y_pred)
    if np.any(y == 0):
        raise ValueError("MAPE is undefined when an actual value is 0")
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100)


def coverage(y_true, low, high) -> float:
    """Share of rows with low <= actual <= high (both edges included), from 0 to 1."""
    y, lo, hi = _arrays(y_true, low, high)
    return float(np.mean((lo <= y) & (y <= hi)))
