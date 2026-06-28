"""Score -> probability calibration and evaluation metrics.

The cortex emits a raw spectral risk in [0,1], not a probability. Platt scaling
fits sigmoid(a*risk + b) to realized default labels so the output is an actual,
calibrated PD. Metrics cover discrimination (AUC, KS) and calibration
(Brier, ECE) so champion vs challenger is judged honestly.
"""
from __future__ import annotations

import numpy as np


def fit_platt(scores: np.ndarray, labels: np.ndarray, iters: int = 500, lr: float = 0.1):
    """Fit PD = sigmoid(a*score + b) by gradient descent. Returns (a, b)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    n = max(len(s), 1)
    for _ in range(iters):
        p = _sigmoid(a * s + b)
        err = p - y
        a -= lr * float(np.dot(err, s) / n)
        b -= lr * float(err.mean())
    return a, b


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * np.asarray(scores, dtype=float) + b)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / rank-based AUC (ties averaged)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks(scores: np.ndarray, labels: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between good/bad score distributions."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(s)
    y = y[order]
    pos_cum = np.cumsum(y) / max(int(y.sum()), 1)
    neg_cum = np.cumsum(1 - y) / max(int((1 - y).sum()), 1)
    return float(np.max(np.abs(pos_cum - neg_cum)))


def brier(pd: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(pd, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def ece(pd: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error over equal-width probability bins."""
    p = np.asarray(pd, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for k in range(bins):
        m = (p >= edges[k]) & (p < edges[k + 1] if k < bins - 1 else p <= edges[k + 1])
        if not m.any():
            continue
        total += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(total)


def default_rate_by_decile(scores: np.ndarray, labels: np.ndarray) -> list[float]:
    """Realized default rate per score decile (low risk -> high risk)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    order = np.argsort(s)
    y = y[order]
    return [float(chunk.mean()) for chunk in np.array_split(y, 10) if len(chunk)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
