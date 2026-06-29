"""Convert SBA 7(a) loan data -> harness export.json (tabular).

SBA loans are small-business credit — the population closest to MCA merchants —
with a clean charge-off label. No transaction series, so (like Lending Club)
this powers the champion + variable-impact, not the spectral challenger.

Get the data:
  Official FOIA: https://data.sba.gov/dataset/7-a-504-foia
  Academic CSV (SBAnational.csv): https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set

Run:
  python research/sba_to_export.py --input SBAnational.csv --out export_sba.json --sample 50000
  python research/train_calibrate_backtest.py --data export_sba.json --out artifacts_sba/

Label: MIS_Status CHGOFF = default(1), P I F = paid(0); others excluded (open).
Harmonized variables (loan_amount, term) pool with Freddie/Lending Club.
"""
from __future__ import annotations

import argparse
import csv
import json

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _f(v, default=0.0) -> float:
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _date(v) -> str:
    s = (v or "").strip()  # "31-Jul-97" or "1997-07-31"
    if "-" in s and len(s.split("-")[0]) <= 2:
        d, mon, yr = s.split("-")
        yr = ("19" + yr) if len(yr) == 2 and int(yr) > 30 else ("20" + yr) if len(yr) == 2 else yr
        return f"{yr}-{MONTHS.get(mon[:3], 1):02d}-{int(d):02d}"
    return s[:10] if len(s) >= 10 else "1990-01-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="export_sba.json")
    ap.add_argument("--sample", type=int, default=50000, help="cap matured loans (0 = all)")
    args = ap.parse_args()

    out = []
    with open(args.input, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            status = (row.get("MIS_Status") or "").strip().upper().replace(" ", "")
            if status == "CHGOFF":
                defaulted = 1
            elif status == "PIF":
                defaulted = 0
            else:
                continue  # open / unknown -> not matured

            gr = _f(row.get("GrAppv") or row.get("DisbursementGross"))
            sba = _f(row.get("SBA_Appv"))
            term = _f(row.get("Term"))
            funded = _date(row.get("DisbursementDate") or row.get("ApprovalDate"))
            out.append({
                "interested_party_id": row.get("LoanNr_ChkDgt") or f"sba-{len(out)}",
                "funded_at": funded,
                "defaulted": defaulted,
                "matured": True,
                "features": {
                    "loan_amount": gr,
                    "term": term,
                    "sba_approved": sba,
                    "sba_guarantee_ratio": sba / gr if gr > 0 else 0.0,
                    "num_employees": _f(row.get("NoEmp")),
                    "created_jobs": _f(row.get("CreateJob")),
                    "retained_jobs": _f(row.get("RetainedJob")),
                    "new_business": 1.0 if _f(row.get("NewExist")) == 2 else 0.0,
                    "urban": 1.0 if _f(row.get("UrbanRural")) == 1 else 0.0,
                    "real_estate_backed": 1.0 if term >= 240 else 0.0,  # known SBA signal
                },
            })
            if args.sample and len(out) >= args.sample:
                break

    json.dump(out, open(args.out, "w"))
    pos = sum(r["defaulted"] for r in out)
    print(f"wrote {len(out)} matured loans ({pos} charge-off / {len(out) - pos} paid, "
          f"{pos / max(len(out), 1) * 100:.1f}% default) to {args.out}")


if __name__ == "__main__":
    main()
