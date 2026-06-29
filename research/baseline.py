"""Champion baseline: a logistic PD over whatever named variables a dataset has.

This is the bar the QuantumCortex challenger must clear. It is dataset-agnostic:
records may carry an explicit ``features`` dict (e.g. Freddie Mac FICO/LTV/DTI);
otherwise it falls back to the MCA summary view (CFR, positions, time-in-business)
so PTM exports and the synthetic path keep working.
"""
from __future__ import annotations

import numpy as np


def record_features(rec: dict) -> dict[str, float]:
    """The tracked quantitative variables for one record.

    Uses rec['features'] when present (any dataset), else derives the MCA view.
    """
    feats = rec.get("features")
    if isinstance(feats, dict) and feats:
        return {k: _num(v) for k, v in feats.items()}

    deposits = _num(rec.get("monthly_deposits_avg"))
    neg = _num(rec.get("negative_days_avg"))
    low = _num(rec.get("low_days_avg"))
    return {
        "cfr": (neg + low) / deposits if deposits > 0 else 0.0,
        "monthly_deposits_avg": deposits,
        "negative_days_avg": neg,
        "low_days_avg": low,
        "current_positions": _num(rec.get("current_positions")),
        "time_in_business_days": _num(rec.get("time_in_business_days")),
    }


def feature_matrix(records: list[dict]) -> tuple[list[str], np.ndarray]:
    """(ordered variable names, X matrix) over records, from record_features."""
    names = list(record_features(records[0]).keys())
    X = np.array([[record_features(r).get(n, 0.0) for n in names] for r in records], dtype=float)
    return names, X


class LogisticBaseline:
    def __init__(self, l2: float = 0.01):
        self.w = None
        self.b = 0.0
        self.mean = None
        self.std = None
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 800, lr: float = 0.1):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9
        Xs = (X - self.mean) / self.std
        n, d = Xs.shape
        self.w = np.zeros(d)
        for _ in range(iters):
            p = _sigmoid(Xs @ self.w + self.b)
            err = p - y
            self.w -= lr * (Xs.T @ err / n + self.l2 * self.w)
            self.b -= lr * err.mean()
        return self

    def predict_pd(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean) / self.std
        return _sigmoid(Xs @ self.w + self.b)


def _num(v, default=0.0) -> float:
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
