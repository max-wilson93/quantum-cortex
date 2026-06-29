"""Quantify how each quantitative variable impacts the default rate.

Dataset-agnostic credit-risk attribution so that whatever source you load
(Freddie Mac, Lending Club, Amex, an internal PTM export), you can see which
variables move default and by how much:

  - Information Value (IV) via Weight-of-Evidence binning — the standard
    credit-scoring measure of a variable's predictive strength.
  - Default rate per quantile bin — the monotonic "does higher X mean more
    default" view, in plain percentages.
  - Point-biserial correlation with the default label (sign + magnitude).

IV rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong,
>0.5 suspiciously strong (check for leakage).
"""
from __future__ import annotations

import numpy as np

from baseline import record_features as variable_for  # dataset-agnostic variables

EPS = 1e-6


def information_value(x: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """IV + per-bin WoE / default-rate table for one variable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0, []  # constant / near-constant variable
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    tot_bad = max(y.sum(), EPS)
    tot_good = max((1 - y).sum(), EPS)

    iv = 0.0
    table = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        bad = float(y[m].sum())
        good = n - bad
        p_bad = max(bad, EPS) / tot_bad
        p_good = max(good, EPS) / tot_good
        woe = float(np.log(p_good / p_bad))
        iv += (p_good - p_bad) * woe
        table.append({
            "bin": b,
            "range": [round(float(edges[b]), 4), round(float(edges[b + 1]), 4)],
            "n": n,
            "default_rate": round(bad / n, 4),
            "woe": round(woe, 4),
        })
    return float(iv), table


def point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < EPS or y.std() < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_impact(records: list[dict]) -> dict:
    """Per-variable IV, correlation, and default-rate-by-bin over the deals."""
    y = np.array([int(r["defaulted"]) for r in records], dtype=float)
    names = list(variable_for(records[0]).keys())
    cols = {n: np.array([variable_for(r)[n] for r in records], dtype=float) for n in names}

    out = []
    for n in names:
        iv, table = information_value(cols[n], y)
        out.append({
            "variable": n,
            "iv": round(iv, 4),
            "strength": _iv_label(iv),
            "corr_with_default": round(point_biserial(cols[n], y), 4),
            "bins": table,
        })
    out.sort(key=lambda d: d["iv"], reverse=True)
    return {"base_default_rate": round(float(y.mean()), 4), "n": len(records), "variables": out}


def print_impact(impact: dict) -> None:
    print(f"\n=== Variable impact on default (base rate "
          f"{impact['base_default_rate'] * 100:.1f}%, n={impact['n']}) ===")
    print(f"{'variable':<24} {'IV':>7}  {'corr':>6}  strength")
    print("-" * 52)
    for v in impact["variables"]:
        print(f"{v['variable']:<24} {v['iv']:>7.3f}  {v['corr_with_default']:>6.2f}  {v['strength']}")


def _iv_label(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "very strong (check leakage)"
