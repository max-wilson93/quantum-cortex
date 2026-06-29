"""Train + calibrate + backtest the QuantumCortex underwriting challenger.

Champion = CFR logistic (baseline.py). Challenger = QuantumCortex spectral PD.
Both are scored on a LEAKAGE-SAFE, origination-date holdout of *matured* deals,
then compared on discrimination + calibration. Emits a versioned artifact the
serving bridge loads: weights.npz + calibration.json + metrics.json.

Usage:
  python research/train_calibrate_backtest.py --data export.json --out artifacts/
  python research/train_calibrate_backtest.py --synth 1200 --out artifacts/   # no real data yet

Dataset (export.json): list of records, each:
  {
    "interested_party_id": "...", "funded_at": "2024-01-15",
    "ledger": [{"posted_at": "2023-...","amount": 1234.5}, ...],   # pre-funding window
    "monthly_deposits_avg": ..., "negative_days_avg": ..., "low_days_avg": ...,
    "current_positions": ..., "time_in_business_days": ...,
    "defaulted": 0|1, "matured": true
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "serving"))

from fourier_optics import FourierOptics  # noqa: E402
from quantum_cortex import QuantumCortex  # noqa: E402
from preprocessing import InsufficientLedgerError, ledger_to_spectral_image  # noqa: E402
from baseline import LogisticBaseline, feature_matrix  # noqa: E402
import calibration as cal  # noqa: E402
import variable_impact as vi  # noqa: E402

MODEL_VERSION = "qc-uw-0.1"
NUM_INPUTS = 3136
NUM_CLASSES = 2  # 0 = healthy, 1 = default  -> risk = P(class 1)
NEURONS = 15
EPOCHS = 3


def build_image(rec: dict, optics: FourierOptics) -> np.ndarray | None:
    ledger = rec.get("ledger")
    if not ledger:  # tabular-only dataset (no time-series) -> no spectral image
        return None
    try:
        img = ledger_to_spectral_image(ledger)
    except InsufficientLedgerError:
        return None
    return optics.apply(img)


def temporal_split(records: list[dict], holdout_frac: float = 0.3):
    matured = [r for r in records if r.get("matured")]
    matured.sort(key=lambda r: r["funded_at"])
    cut = int(len(matured) * (1 - holdout_frac))
    return matured[:cut], matured[cut:]


def train_cortex(train: list[dict], feats: dict[int, np.ndarray]) -> QuantumCortex:
    cortex = QuantumCortex(NUM_INPUTS, NUM_CLASSES, NEURONS)
    idx = [i for i in range(len(train)) if feats.get(i) is not None]
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for n, i in enumerate(idx):
            cortex.process_image(feats[i], int(train[i]["defaulted"]), train=True)
            if (n + 1) % 200 == 0:
                cortex.decay_learning_rate((ep + n / len(idx)) / EPOCHS)
    return cortex


def cortex_risk(cortex: QuantumCortex, feature_vec: np.ndarray) -> float:
    cortex.process_image(feature_vec, label=0, train=False)
    e = np.asarray(cortex.last_class_energies, dtype=float)
    total = e.sum()
    return float(e[1] / total) if total > 0 else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to export.json")
    ap.add_argument("--synth", type=int, default=0, help="generate N synthetic deals instead")
    ap.add_argument("--out", default="artifacts", help="artifact output dir")
    args = ap.parse_args()

    records = _synth(args.synth) if args.synth else json.load(open(args.data))
    train, test = temporal_split(records)
    if not train or not test:
        raise SystemExit("need matured deals in both train and holdout windows")
    print(f"matured: {len(train)} train / {len(test)} holdout (origination-date split)")

    # --- Champion + attribution: ALL matured rows (no time-series required) so
    #     tabular-rich datasets (e.g. Lending Club) are used to the fullest. ---
    feat_names, Xtr = feature_matrix(train)
    _, Xte = feature_matrix(test)
    tr_y = np.array([int(r["defaulted"]) for r in train], dtype=float)
    te_y = np.array([int(r["defaulted"]) for r in test], dtype=float)
    base = LogisticBaseline().fit(Xtr, tr_y)
    cp_pd = base.predict_pd(Xte)

    metrics = {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_train": len(train), "n_holdout": len(test),
        "holdout_default_rate": float(te_y.mean()),
        "champion_logistic": _eval(cp_pd, te_y),
    }

    # --- Challenger: spectral PD, ONLY over rows that have a usable ledger. ---
    optics = FourierOptics(shape=(28, 28))
    feats_tr = {i: build_image(r, optics) for i, r in enumerate(train)}
    feats_te = {i: build_image(r, optics) for i, r in enumerate(test)}
    tr_idx = [i for i in feats_tr if feats_tr[i] is not None]
    te_idx = [i for i in feats_te if feats_te[i] is not None]

    cortex, a, b = None, 1.0, 0.0
    if len(tr_idx) >= 30 and len(te_idx) >= 10:
        cortex = train_cortex(train, feats_tr)
        tr_risk = np.array([cortex_risk(cortex, feats_tr[i]) for i in tr_idx])
        a, b = cal.fit_platt(tr_risk, np.array([int(train[i]["defaulted"]) for i in tr_idx]))
        cte_y = np.array([int(test[i]["defaulted"]) for i in te_idx])
        ch_pd = cal.apply_platt(np.array([cortex_risk(cortex, feats_te[i]) for i in te_idx]), a, b)
        metrics["n_holdout_timeseries"] = len(te_idx)
        metrics["challenger_quantum_spectral"] = _eval(ch_pd, cte_y)
        metrics["promote_challenger"] = bool(
            metrics["challenger_quantum_spectral"]["auc"] > metrics["champion_logistic"]["auc"]
        )
    else:
        metrics["challenger_quantum_spectral"] = None  # tabular-only dataset
        metrics["promote_challenger"] = False
        print("No usable time-series in this dataset -> spectral challenger skipped "
              "(tabular-only). Champion + variable impact still produced.")

    # Variable-impact attribution over all matured rows + champion's signed weights.
    impact = vi.compute_impact(train + test)
    impact["champion_logistic_weights"] = {
        n: round(float(w), 4) for n, w in zip(feat_names, base.w)
    }
    metrics["variable_impact"] = impact

    os.makedirs(args.out, exist_ok=True)
    json.dump(impact, open(os.path.join(args.out, "variable_impact.json"), "w"), indent=2)
    if cortex is not None:  # only a time-series dataset yields a servable artifact
        np.savez(os.path.join(args.out, "weights.npz"), W_in=cortex.W_in, W_lat=cortex.W_lat)
        json.dump(
            {"model_version": MODEL_VERSION, "num_classes": NUM_CLASSES,
             "platt": {"a": a, "b": b}},
            open(os.path.join(args.out, "calibration.json"), "w"), indent=2,
        )
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "variable_impact"}, indent=2))
    vi.print_impact(impact)
    artifacts = "metrics.json, variable_impact.json" + (
        ", weights.npz, calibration.json" if cortex is not None else ""
    )
    print(f"\nArtifact written to {args.out}/ ({artifacts})")
    print("Promote challenger:", metrics["promote_challenger"])


def _eval(pd: np.ndarray, y: np.ndarray) -> dict:
    return {
        "auc": round(cal.auc(pd, y), 4),
        "ks": round(cal.ks(pd, y), 4),
        "brier": round(cal.brier(pd, y), 4),
        "ece": round(cal.ece(pd, y), 4),
        "default_rate_by_decile": [round(x, 3) for x in cal.default_rate_by_decile(pd, y)],
    }


def _synth(n: int) -> list[dict]:
    """Synthetic deals with a real signal: distress shows up as a low-frequency
    downward drift + spikiness in the ledger, which the spectral model can see
    and monthly averages partly miss. Lets you exercise the full pipeline now."""
    rng = np.random.default_rng(7)
    out = []
    base_day = np.datetime64("2023-01-01")
    for k in range(n):
        distressed = rng.random() < 0.25
        days = np.sort(rng.choice(np.arange(0, 300), size=rng.integers(40, 120), replace=False))
        trend = -0.04 if distressed else 0.01
        amp = 1.8 if distressed else 0.8
        amount = (
            500
            + 50 * np.sin(2 * np.pi * days / (12 if distressed else 30))
            * amp
            + trend * days * 30
            + rng.normal(0, 60 * amp, size=len(days))
        )
        ledger = [
            {"posted_at": str(base_day + int(d)), "amount": float(v)}
            for d, v in zip(days, amount)
        ]
        out.append({
            "interested_party_id": f"synth-{k}",
            "funded_at": str(base_day + int(days[-1]) + 5),
            "ledger": ledger,
            "monthly_deposits_avg": float(max(amount.mean() * 8, 1)),
            "negative_days_avg": float(6 if distressed else 1) + rng.random(),
            "low_days_avg": float(9 if distressed else 2) + rng.random(),
            "current_positions": int(rng.integers(2, 5) if distressed else rng.integers(0, 2)),
            "time_in_business_days": int(rng.integers(200, 700) if distressed else rng.integers(700, 3000)),
            "defaulted": int(distressed if rng.random() < 0.85 else (not distressed)),
            "matured": True,
        })
    return out


if __name__ == "__main__":
    main()
