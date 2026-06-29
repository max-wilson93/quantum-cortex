"""Cross-dataset variable-impact: pool shared variables for a bigger sample.

Loads several export.json files and, for each quantitative variable, reports its
impact on default *per dataset* and *pooled* across the datasets that contain it,
plus whether the direction is consistent. A variable that moves default the same
way across mortgages (Freddie), consumer loans (Lending Club), and small business
(SBA) is a robust signal worth trusting in the MCA model.

Usage:
  python research/pool_impact.py \
    --data export_freddie.json export_lc.json export_sba.json \
    --names freddie lending_club sba --out pooled_impact.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from baseline import record_features
import variable_impact as vi


def _col(records: list[dict], var: str):
    x = np.array([record_features(r).get(var, np.nan) for r in records], dtype=float)
    y = np.array([int(r["defaulted"]) for r in records], dtype=float)
    m = ~np.isnan(x)
    return x[m], y[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--out", default="pooled_impact.json")
    args = ap.parse_args()

    names = args.names or [f"ds{i}" for i in range(len(args.data))]
    datasets = {n: json.load(open(p)) for n, p in zip(names, args.data)}
    var_names = {n: list(record_features(recs[0]).keys()) for n, recs in datasets.items()}
    all_vars = sorted({v for vs in var_names.values() for v in vs})

    results = []
    for var in all_vars:
        present = [n for n in names if var in var_names[n]]
        per = {}
        pooled_x, pooled_y = [], []
        for n in present:
            x, y = _col(datasets[n], var)
            if len(x) < 20 or x.std() < 1e-9:
                continue
            iv, _ = vi.information_value(x, y)
            per[n] = {"iv": round(iv, 4), "corr": round(vi.point_biserial(x, y), 4), "n": int(len(x))}
            pooled_x.append(x)
            pooled_y.append(y)
        if not per:
            continue
        entry = {"variable": var, "datasets": list(per.keys()), "per_dataset": per}
        if len(per) >= 2:
            X = np.concatenate(pooled_x)
            Y = np.concatenate(pooled_y)
            piv, _ = vi.information_value(X, Y)
            entry["pooled"] = {"iv": round(piv, 4), "corr": round(vi.point_biserial(X, Y), 4),
                               "n": int(len(X))}
            signs = {np.sign(d["corr"]) for d in per.values() if d["corr"] != 0}
            entry["direction_consistent"] = len(signs) <= 1
        results.append(entry)

    shared = [e for e in results if "pooled" in e]
    shared.sort(key=lambda e: e["pooled"]["iv"], reverse=True)
    single = [e for e in results if "pooled" not in e]

    json.dump({"datasets": {n: len(r) for n, r in datasets.items()},
               "shared_variables": shared, "single_dataset_variables": single},
              open(args.out, "w"), indent=2)

    print(f"datasets: " + ", ".join(f"{n}={len(r)}" for n, r in datasets.items()))
    print(f"\n=== SHARED variables (pooled across >=2 datasets) ===")
    print(f"{'variable':<22} {'pooled IV':>9} {'pooled corr':>12} {'consistent':>11}  per-dataset corr")
    print("-" * 90)
    for e in shared:
        pc = "  ".join(f"{n}:{d['corr']:+.2f}" for n, d in e["per_dataset"].items())
        print(f"{e['variable']:<22} {e['pooled']['iv']:>9.3f} {e['pooled']['corr']:>12.2f} "
              f"{str(e['direction_consistent']):>11}  {pc}")
    if single:
        print(f"\nsingle-dataset variables: " + ", ".join(e["variable"] for e in single))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
