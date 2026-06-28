"""Turn any 'transactions + outcome label' source into the harness export.json.

This is how we feed *relevant* data to the model from many sources through one
path. Point it at:
  - an internal PTM export (manual ledgers / parsed bank statements + deal_outcomes), or
  - a public proxy dataset with a transaction time-series + a good/bad loan label.

Inputs are two CSVs:
  transactions.csv : account_id, date, amount         (the time-series / features)
  labels.csv       : account_id, defaulted[, funded_at, monthly_deposits_avg,
                     negative_days_avg, low_days_avg, current_positions,
                     time_in_business_days]            (the supervised label + summary)

Column names are configurable so the same tool maps different schemas.

Usage:
  python research/ingest_dataset.py \
    --transactions tx.csv --labels y.csv --out export.json \
    --tx-account account_id --tx-date date --tx-amount amount \
    --lab-account account_id --lab-default defaulted
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict


def _num(v, default=None):
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="export.json")
    ap.add_argument("--tx-account", default="account_id")
    ap.add_argument("--tx-date", default="date")
    ap.add_argument("--tx-amount", default="amount")
    ap.add_argument("--lab-account", default="account_id")
    ap.add_argument("--lab-default", default="defaulted")
    ap.add_argument("--lab-funded-at", default="funded_at")
    ap.add_argument("--min-txns", type=int, default=16)
    args = ap.parse_args()

    # 1. Group transactions per account.
    ledgers: dict[str, list[dict]] = defaultdict(list)
    with open(args.transactions, newline="") as f:
        for row in csv.DictReader(f):
            acct = row.get(args.tx_account)
            amt = _num(row.get(args.tx_amount))
            date = row.get(args.tx_date)
            if acct is None or amt is None or not date:
                continue
            ledgers[acct].append({"posted_at": date, "amount": amt})

    # 2. Attach labels + optional summary features; emit export.json records.
    out = []
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f):
            acct = row.get(args.lab_account)
            led = ledgers.get(acct)
            if not led or len(led) < args.min_txns:
                continue
            led.sort(key=lambda r: r["posted_at"])
            defaulted = row.get(args.lab_default)
            if defaulted is None or str(defaulted).strip() == "":
                continue
            out.append({
                "interested_party_id": acct,
                "funded_at": row.get(args.lab_funded_at) or led[-1]["posted_at"],
                "ledger": led,
                # Summary features for the CFR champion; default to neutral if absent.
                "monthly_deposits_avg": _num(row.get("monthly_deposits_avg"), 0.0),
                "negative_days_avg": _num(row.get("negative_days_avg"), 0.0),
                "low_days_avg": _num(row.get("low_days_avg"), 0.0),
                "current_positions": _num(row.get("current_positions"), 0.0),
                "time_in_business_days": _num(row.get("time_in_business_days"), 0.0),
                "defaulted": int(float(str(defaulted))),
                "matured": True,
            })

    json.dump(out, open(args.out, "w"))
    pos = sum(r["defaulted"] for r in out)
    print(f"wrote {len(out)} records to {args.out} "
          f"({pos} default / {len(out) - pos} healthy, "
          f"{pos / len(out) * 100:.1f}% default rate)" if out else "no usable records")


if __name__ == "__main__":
    main()
