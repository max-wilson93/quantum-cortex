"""Convert PaySim synthetic mobile-money data -> harness export.json.

IMPORTANT — read before using:
  * PaySim's label is `isFraud`, NOT loan default. This converter trains a FRAUD
    proxy. Use it to validate that the Lomb-Scargle -> cortex spectral pipeline
    learns signal from real transaction streams, and as a scale test — NOT to
    calibrate MCA default PD, and NOT to pool with the default datasets.
  * Per-account histories are short/sparse, so many accounts won't reach the
    ~16 distinct-day minimum the spectral stage needs; expect a modest number of
    usable samples out of millions of rows.

Get the data: https://www.kaggle.com/datasets/ealaxi/paysim1  (PS_*.csv, ~6.3M rows)

Run:
  python research/paysim_to_export.py --input paysim.csv --out export_paysim.json \
    --min-txns 16 --max-rows 2000000
  python research/train_calibrate_backtest.py --data export_paysim.json --out artifacts_paysim/
"""
from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from datetime import datetime, timedelta
import json

BASE = datetime(2023, 1, 1)
OUTFLOW = {"CASH_OUT", "PAYMENT", "DEBIT", "TRANSFER"}


def _open(path: str):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def _signed(txn_type: str, amount: float) -> float:
    return -amount if txn_type in OUTFLOW else amount  # CASH_IN = inflow (+)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="export_paysim.json")
    ap.add_argument("--min-txns", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows ingested (0 = all)")
    args = ap.parse_args()

    ledgers: dict[str, list] = defaultdict(list)
    fraud: dict[str, int] = defaultdict(int)

    with _open(args.input) as f:
        for n, row in enumerate(csv.DictReader(f)):
            if args.max_rows and n >= args.max_rows:
                break
            acct = row.get("nameOrig")
            if not acct:
                continue
            try:
                step = int(row["step"]); amount = float(row["amount"])
            except (KeyError, ValueError):
                continue
            ledgers[acct].append((step, _signed(row.get("type", ""), amount)))
            if row.get("isFraud") in ("1", "1.0", 1):
                fraud[acct] = 1

    out = []
    for acct, txns in ledgers.items():
        if len(txns) < args.min_txns:
            continue
        txns.sort()
        ledger = [{"posted_at": (BASE + timedelta(hours=s)).isoformat(), "amount": round(a, 2)}
                  for s, a in txns]
        amounts = [a for _, a in txns]
        out_sum = -sum(a for a in amounts if a < 0)
        in_sum = sum(a for a in amounts if a > 0)
        out.append({
            "interested_party_id": acct,
            "funded_at": ledger[-1]["posted_at"][:10],
            "ledger": ledger,
            "defaulted": int(fraud.get(acct, 0)),  # FRAUD proxy, not default
            "matured": True,
            "features": {
                "n_txns": float(len(txns)),
                "total_out": round(out_sum, 2),
                "total_in": round(in_sum, 2),
                "mean_amount": round(sum(abs(a) for a in amounts) / len(amounts), 2),
                "out_in_ratio": round(out_sum / in_sum, 4) if in_sum > 0 else 0.0,
            },
        })

    json.dump(out, open(args.out, "w"))
    pos = sum(r["defaulted"] for r in out)
    print(f"accounts>= {args.min_txns} txns: {len(out)} "
          f"({pos} fraud / {len(out) - pos} clean) -> {args.out}")
    print("NOTE: label = isFraud (not default). Spectral-pipeline validation only;"
          " do NOT pool with default datasets.")


if __name__ == "__main__":
    main()
