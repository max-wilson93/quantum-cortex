#!/usr/bin/env bash
# Chain convert -> train -> pool for whichever datasets you've downloaded.
#
# Point the env vars at your files (any subset works; missing ones are skipped):
#   FREDDIE_ORIG / FREDDIE_PERF  Freddie Mac origination + performance files
#   LC_INPUT                     Lending Club accepted_*.csv[.gz]
#   SBA_INPUT                    SBA SBAnational.csv
#   OUT                          output dir (default: ./artifacts)
#   SAMPLE                       per-dataset loan cap (default: 8000)
#
# Example:
#   FREDDIE_ORIG=historical_data_2018Q1.txt \
#   FREDDIE_PERF=historical_data_time_2018Q1.txt \
#   LC_INPUT=accepted_2007_to_2018Q4.csv.gz \
#   SBA_INPUT=SBAnational.csv \
#   bash research/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (so research/* and engine imports resolve)

OUT="${OUT:-artifacts}"
SAMPLE="${SAMPLE:-8000}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"

exports=()   # export.json files produced
names=()     # matching dataset names

run_train() {  # $1=export.json  $2=name
  echo ">>> train/backtest: $2"
  "$PY" research/train_calibrate_backtest.py --data "$1" --out "$OUT/$2" || return 1
  exports+=("$1"); names+=("$2")
}

# --- Freddie Mac (time-series: trains the spectral model) ---
if [[ -n "${FREDDIE_ORIG:-}" && -n "${FREDDIE_PERF:-}" && -f "${FREDDIE_ORIG}" && -f "${FREDDIE_PERF}" ]]; then
  echo ">>> convert: freddie"
  "$PY" research/freddie_to_export.py --origination "$FREDDIE_ORIG" \
    --performance "$FREDDIE_PERF" --out "$OUT/export_freddie.json" --sample "$SAMPLE" \
    && run_train "$OUT/export_freddie.json" freddie
else
  echo "-- skip freddie (set FREDDIE_ORIG + FREDDIE_PERF to existing files)"
fi

# --- Lending Club (tabular: rich attribution) ---
if [[ -n "${LC_INPUT:-}" && -f "${LC_INPUT}" ]]; then
  echo ">>> convert: lending_club"
  "$PY" research/lendingclub_to_export.py --input "$LC_INPUT" \
    --out "$OUT/export_lc.json" --sample $((SAMPLE * 5)) \
    && run_train "$OUT/export_lc.json" lending_club
else
  echo "-- skip lending_club (set LC_INPUT to an existing file)"
fi

# --- SBA 7(a) (tabular: small-business population) ---
if [[ -n "${SBA_INPUT:-}" && -f "${SBA_INPUT}" ]]; then
  echo ">>> convert: sba"
  "$PY" research/sba_to_export.py --input "$SBA_INPUT" \
    --out "$OUT/export_sba.json" --sample $((SAMPLE * 5)) \
    && run_train "$OUT/export_sba.json" sba
else
  echo "-- skip sba (set SBA_INPUT to an existing file)"
fi

# --- Pooled cross-dataset attribution (needs >= 2 datasets) ---
if (( ${#exports[@]} >= 2 )); then
  echo ">>> pool: ${names[*]}"
  "$PY" research/pool_impact.py --data "${exports[@]}" --names "${names[@]}" \
    --out "$OUT/pooled_impact.json"
elif (( ${#exports[@]} == 1 )); then
  echo "-- only one dataset; per-dataset impact in $OUT/${names[0]}/variable_impact.json"
else
  echo "!! no datasets processed — set at least one of FREDDIE_*/LC_INPUT/SBA_INPUT"
  exit 1
fi

echo
echo "Done. Outputs in $OUT/:"
echo "  per-dataset: <name>/metrics.json, <name>/variable_impact.json (+ weights.npz for freddie)"
(( ${#exports[@]} >= 2 )) && echo "  pooled:      pooled_impact.json"
