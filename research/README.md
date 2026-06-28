# Underwriting model: calibrate / train / backtest

Offline harness that turns the QuantumCortex into a *specific* MCA underwriting
model and proves (or disproves) it against a baseline before it prices anything.

## Design
- **Target:** a calibrated **probability of default (PD)**, not a factor rate.
  Pricing (PD → LGD → expected loss → grade → terms) stays PTM's job (`rateDeal`).
- **Champion:** CFR logistic PD from monthly summary stats (`baseline.py`) —
  mirrors PTM's existing engine.
- **Challenger:** QuantumCortex spectral PD — Lomb-Scargle image of the raw
  ledger → cortex → calibrated PD. Its only possible edge is temporal/frequency
  structure that monthly averages discard.
- **Gate:** the challenger is promoted **only if it beats the champion AUC** on a
  leakage-safe, origination-date holdout of *matured* deals.

## Run
```bash
# No real labels yet — exercise the whole pipeline on synthetic deals:
python research/train_calibrate_backtest.py --synth 1200 --out artifacts/

# With a real export (deal_outcomes ⨝ interested_parties ⨝ daily_ledger_sync):
python research/train_calibrate_backtest.py --data export.json --out artifacts/
```

## Output artifact (consumed by the serving bridge)
```
artifacts/
  weights.npz        # trained W_in / W_lat
  calibration.json   # num_classes + Platt (raw risk -> calibrated PD) + model_version
  metrics.json       # champion vs challenger: AUC, KS, Brier, ECE, decile default rates
```
Point the bridge at it with `CORTEX_ARTIFACT_DIR`. The StatefulSet expects the
training Job to write the artifact to the shared `/data/artifacts` volume.

## Honest note on the synthetic run
On `--synth` data the champion wins (the baked-in signal is also visible to the
summary stats), so `promote_challenger=false`. That is the harness working as
intended: it will not promote the spectral model unless real ledger data gives
it a genuine, measured edge.

## Labels are the prerequisite
Training needs `deal_outcomes` rows (defaulted / collected / matured). Until
those accrue, run `--synth` to validate plumbing; the model stays untrained (the
bridge falls back to identity calibration) and PTM keeps using the CFR engine.
```
export.json record:
  { "interested_party_id", "funded_at", "ledger":[{"posted_at","amount"}...],
    "monthly_deposits_avg","negative_days_avg","low_days_avg",
    "current_positions","time_in_business_days", "defaulted":0|1, "matured":true }
```
