"""Convert Freddie Mac Single-Family Loan-Level data -> harness export.json.

Why Freddie: each loan has a long MONTHLY performance series (what the spectral
model needs) plus named, interpretable variables (FICO, LTV, DTI, rate) for the
variable-impact report. Current data, updated to the present.

Get the data (free registration):
  https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
You download two pipe-delimited files per vintage:
  historical_data_YYYYQn.txt        (origination)
  historical_data_time_YYYYQn.txt   (monthly performance)

Run:
  python research/freddie_to_export.py \
    --origination historical_data_2018Q1.txt \
    --performance historical_data_time_2018Q1.txt \
    --out export_freddie.json --sample 8000
  python research/train_calibrate_backtest.py --data export_freddie.json --out artifacts/

Field indices follow the Standard dataset layout; adjust the constants below if
your vintage differs. Default = 90+ days delinquent ever, or REO.
"""
from __future__ import annotations

import argparse
import json

# --- Standard layout column indices (0-based) ---
# origination
O_FICO, O_FIRST_PMT, O_CLTV, O_DTI, O_UPB, O_LTV, O_RATE, O_LOANSEQ = 0, 1, 8, 9, 10, 11, 12, 19
# performance
P_LOANSEQ, P_PERIOD, P_UPB, P_DELQ, P_AGE, P_ZBC = 0, 1, 2, 3, 4, 8

DEFAULT_DELQ = 3  # >= 3 monthly reporting => 90+ DPD => default proxy


def _f(v: str, hi: float | None = None) -> float:
    """Parse a numeric field; Freddie uses 9999/999 for unknown -> 0.0."""
    s = (v or "").strip()
    try:
        x = float(s)
    except ValueError:
        return 0.0
    if hi is not None and x >= hi:
        return 0.0  # unknown sentinel
    return x


def _period_to_date(yyyymm: str) -> str:
    s = (yyyymm or "").strip()
    return f"{s[:4]}-{s[4:6]}-01" if len(s) >= 6 else "1970-01-01"


def _is_default(delq: str) -> bool:
    s = (delq or "").strip()
    if s in ("R",):  # REO
        return True
    try:
        return int(s) >= DEFAULT_DELQ
    except ValueError:
        return False  # 'XX' / blank


def load_origination(path: str, sample: int | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            c = line.rstrip("\n").split("|")
            if len(c) <= O_LOANSEQ:
                continue
            seq = c[O_LOANSEQ].strip()
            out[seq] = {
                "features": {
                    "fico": _f(c[O_FICO], hi=9999),
                    "orig_ltv": _f(c[O_LTV], hi=999),
                    "orig_cltv": _f(c[O_CLTV], hi=999),
                    "orig_dti": _f(c[O_DTI], hi=999),
                    "orig_interest_rate": _f(c[O_RATE]),
                    "orig_upb": _f(c[O_UPB]),
                },
                "funded_at": _period_to_date(c[O_FIRST_PMT]),
                "orig_upb": _f(c[O_UPB]),
            }
            if sample and len(out) >= sample:
                break
    return out


def attach_performance(path: str, loans: dict[str, dict]) -> list[dict]:
    series: dict[str, list[tuple[str, float, bool, bool]]] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            c = line.rstrip("\n").split("|")
            if len(c) <= P_DELQ:
                continue
            seq = c[P_LOANSEQ].strip()
            if seq not in loans:
                continue
            zbc = c[P_ZBC].strip() if len(c) > P_ZBC else ""
            series.setdefault(seq, []).append(
                (_period_to_date(c[P_PERIOD]), _f(c[P_UPB]), _is_default(c[P_DELQ]), bool(zbc))
            )

    records = []
    for seq, rows in series.items():
        rows.sort(key=lambda r: r[0])
        meta = loans[seq]
        # Ledger = monthly principal paid (prev_upb - upb), dated at the period.
        ledger, prev = [], meta["orig_upb"]
        for date, upb, _d, _z in rows:
            ledger.append({"posted_at": date, "amount": round(prev - upb, 2)})
            prev = upb
        defaulted = any(d for _, _, d, _ in rows)
        terminated = any(z for _, _, _, z in rows)
        records.append({
            "interested_party_id": seq,
            "funded_at": meta["funded_at"],
            "features": meta["features"],
            "ledger": ledger,
            "defaulted": int(defaulted),
            # Resolved outcome only (terminated or already defaulted) => no look-ahead.
            "matured": bool(terminated or defaulted),
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origination", required=True)
    ap.add_argument("--performance", required=True)
    ap.add_argument("--out", default="export_freddie.json")
    ap.add_argument("--sample", type=int, default=8000, help="cap loans (0 = all)")
    args = ap.parse_args()

    loans = load_origination(args.origination, args.sample or None)
    records = attach_performance(args.performance, loans)
    matured = [r for r in records if r["matured"]]
    json.dump(records, open(args.out, "w"))

    pos = sum(r["defaulted"] for r in matured)
    print(f"loans={len(loans)} records={len(records)} matured={len(matured)} "
          f"defaults={pos} ({pos / max(len(matured),1) * 100:.1f}% of matured)")
    print(f"wrote {args.out} -> feed to train_calibrate_backtest.py --data {args.out}")


if __name__ == "__main__":
    main()
