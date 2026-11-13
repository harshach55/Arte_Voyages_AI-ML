"""Baseline: median adr per hotel + arrival month + reserved room type, from the training data.

If a group was never seen in training, fall back to hotel + month, then hotel, then the
global median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = [
    ["hotel", "arrival_month", "reserved_room_type"],
    ["hotel", "arrival_month"],
    ["hotel"],
]
LEVEL_NAMES = ["hotel+month+room", "hotel+month", "hotel", "global"]


class MedianBaseline:
    def fit(self, X: pd.DataFrame, y) -> "MedianBaseline":
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.tables_ = []
        for cols in LEVELS:
            keys = [X[c] for c in cols]
            self.tables_.append(y.groupby(keys).median())
        self.global_median_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame, return_level: bool = False):
        out = np.full(len(X), np.nan)
        level = np.full(len(X), len(LEVELS), dtype=int)
        for i, (cols, table) in enumerate(zip(LEVELS, self.tables_)):
            todo = np.isnan(out)
            if not todo.any():
                break
            part = X.loc[todo, cols]
            keys = pd.MultiIndex.from_frame(part) if len(cols) > 1 else pd.Index(part[cols[0]])
            values = table.reindex(keys).to_numpy(dtype=float)
            found = ~np.isnan(values)
            idx = np.flatnonzero(todo)
            out[idx[found]] = values[found]
            level[idx[found]] = i
        out[np.isnan(out)] = self.global_median_
        if return_level:
            return out, [LEVEL_NAMES[i] for i in level]
        return out
