"""Champion baseline: a CFR-style logistic PD, mirroring PTM's rateDeal().

This is the bar the QuantumCortex challenger must clear. It uses the same
summary features PTM's first-principles engine uses (cash-flow ratio, positions,
time-in-business) so the backtest isolates the value of spectral processing.
"""
from __future__ import annotations

import numpy as np


def cfr_features(row: dict) -> np.ndarray:
    """Summary features from a deal record (the monthly-average view)."""
    deposits = float(row.get("monthly_deposits_avg") or 0)
    neg_days = float(row.get("negative_days_avg") or 0)
    low_days = float(row.get("low_days_avg") or 0)
    cfr = (neg_days + low_days) / deposits if deposits > 0 else 0.0
    positions = float(row.get("current_positions") or 0)
    tib_days = float(row.get("time_in_business_days") or 0)
    return np.array([cfr, positions, tib_days * 1e-3], dtype=float)


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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
