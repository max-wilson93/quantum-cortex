"""Convert Lending Club accepted-loans data -> harness export.json (tabular).

Lending Club is the richest set of *interpretable* borrower variables (income,
DTI, FICO, utilization, inquiries, delinquencies...) but has NO per-loan
transaction series — so it powers the champion baseline + variable-impact
attribution, not the spectral challenger (the harness detects the missing ledger
and skips the cortex automatically).

Get the data (Kaggle, free account):
  https://www.kaggle.com/datasets/wordsforthewise/lending-club
  (accepted_2007_to_2018Q4.csv.gz)

Run:
  python research/lendingclub_to_export.py --input accepted_2007_to_2018Q4.csv.gz \
    --out export_lc.json --sample 40000
  python research/train_calibrate_backtest.py --data export_lc.json --out artifacts/
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json

BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _f(v, default=0.0) -> float:
    try:
        return float(str(v).replace("%", "").replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _emp_years(v) -> float:
    s = (v or "").strip()
    if s.startswith("<"):
        return 0.0
    if "10" in s:
        return 10.0
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else 0.0


def _issue_to_date(v) -> str:
    s = (v or "").strip()  # e.g. "Dec-2015"
    if "-" in s:
        mon, yr = s.split("-")
        return f"{yr}-{MONTHS.get(mon[:3], 1):02d}-01"
    return "1970-01-01"


def _months_between(earliest: str, issue: str) -> float:
    try:
        em, ey = earliest.split("-"); im, iy = issue.split("-")
        return (int(iy) - int(ey)) * 12 + (MONTHS.get(im[:3], 1) - MONTHS.get(em[:3], 1))
    except Exception:
        return 0.0


def _open(path: str):
    return gzip.open(path, "rt", encoding="latin-1") if path.endswith(".gz") \
        else open(path, encoding="latin-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="export_lc.json")
    ap.add_argument("--sample", type=int, default=40000, help="cap matured loans (0 = all)")
    args = ap.parse_args()

    out = []
    with _open(args.input) as f:
        for row in csv.DictReader(f):
            status = (row.get("loan_status") or "").strip()
            if status in BAD:
                defaulted = 1
            elif status in GOOD:
                defaulted = 0
            else:
                continue  # Current / Late / In Grace = not matured -> exclude

            issue = _issue_to_date(row.get("issue_d"))
            fico = (_f(row.get("fico_range_low")) + _f(row.get("fico_range_high"))) / 2
            inc = _f(row.get("annual_inc"))
            installment = _f(row.get("installment"))
            term = _f("".join(ch for ch in (row.get("term") or "") if ch.isdigit()))
            out.append({
                "interested_party_id": row.get("id") or f"lc-{len(out)}",
                "funded_at": issue,
                "defaulted": defaulted,
                "matured": True,
                "features": {
                    "fico": fico,
                    "annual_inc": inc,
                    "dti": _f(row.get("dti")),
                    "int_rate": _f(row.get("int_rate")),
                    "revol_util": _f(row.get("revol_util")),
                    "revol_bal": _f(row.get("revol_bal")),
                    "loan_amount": _f(row.get("loan_amnt")),  # harmonized name (pools)
                    "term": term,  # " 36 months" -> 36
                    "installment": installment,
                    "open_acc": _f(row.get("open_acc")),
                    "total_acc": _f(row.get("total_acc")),
                    "inq_last_6mths": _f(row.get("inq_last_6mths")),
                    "delinq_2yrs": _f(row.get("delinq_2yrs")),
                    "pub_rec": _f(row.get("pub_rec")),
                    "emp_length_years": _emp_years(row.get("emp_length")),
                    "credit_history_months": _months_between(
                        _issue_to_date(row.get("earliest_cr_line")), issue),
                    # engineered enrichment
                    "installment_to_income": installment * 12 / inc if inc > 0 else 0.0,
                },
            })
            if args.sample and len(out) >= args.sample:
                break

    json.dump(out, open(args.out, "w"))
    pos = sum(r["defaulted"] for r in out)
    print(f"wrote {len(out)} matured loans ({pos} default / {len(out) - pos} paid, "
          f"{pos / max(len(out), 1) * 100:.1f}% default) to {args.out}")
    print(f"feed: python research/train_calibrate_backtest.py --data {args.out} --out artifacts/")


if __name__ == "__main__":
    main()
