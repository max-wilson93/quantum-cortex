# Training data for the underwriting model

## One command (after you've downloaded any subset of the files)
```bash
FREDDIE_ORIG=historical_data_2018Q1.txt FREDDIE_PERF=historical_data_time_2018Q1.txt \
LC_INPUT=accepted_2007_to_2018Q4.csv.gz SBA_INPUT=SBAnational.csv \
bash research/run_all.sh        # convert -> train each -> pooled attribution
```
Any subset works (missing datasets are skipped). Outputs land in `artifacts/`:
per-dataset `metrics.json` + `variable_impact.json` (+ `weights.npz` for Freddie),
and `pooled_impact.json` when >=2 datasets are present.

## Train today (Freddie Mac — fastest real run)
Each loan has a long monthly series (for the spectral model) + named variables
(for attribution). Steps:
1. Register (free) and download one vintage's two pipe-delimited files from
   https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset —
   `historical_data_YYYYQn.txt` (origination) + `historical_data_time_YYYYQn.txt`
   (performance).
2. Convert + train:
   ```bash
   pip install -r serving/requirements.txt   # numpy/scipy/scikit-image
   python research/freddie_to_export.py \
     --origination historical_data_2018Q1.txt \
     --performance historical_data_time_2018Q1.txt \
     --out export_freddie.json --sample 8000
   python research/train_calibrate_backtest.py --data export_freddie.json --out artifacts/
   ```
3. Read `artifacts/metrics.json` (champion vs challenger) and
   `artifacts/variable_impact.json` (how FICO/LTV/DTI move default). The trained
   `weights.npz`+`calibration.json` are what the serving bridge loads.

Caveat: mortgages, not MCA — this validates the *method* and the variable-impact
tooling on real, current data. Swap in your internal `deal_outcomes` export
(Tier 1) the moment labels accrue.

## Enrich attribution with Lending Club (tabular, no registration friction)
Lending Club has the richest *interpretable* variables (income, DTI, FICO,
utilization, inquiries, delinquencies, ...) but no per-loan series — so it powers
the **champion + variable-impact**, and the harness auto-skips the spectral
challenger (tabular-only).
```bash
# Kaggle: wordsforthewise/lending-club -> accepted_2007_to_2018Q4.csv.gz
python research/lendingclub_to_export.py --input accepted_2007_to_2018Q4.csv.gz \
  --out export_lc.json --sample 40000
python research/train_calibrate_backtest.py --data export_lc.json --out artifacts_lc/
```
Use it to cross-check which variables drive default against Freddie + your
internal deals (16 variables incl. engineered installment-to-income).

## SBA 7(a) — small business, closest population to MCA (tabular)
Small-business loans with a charge-off label — the borrower population nearest to
MCA merchants. Tabular (champion + attribution only).
```bash
# data.sba.gov/dataset/7-a-504-foia  OR Kaggle larsen0966/sba-loans-case-data-set
python research/sba_to_export.py --input SBAnational.csv --out export_sba.json --sample 50000
python research/train_calibrate_backtest.py --data export_sba.json --out artifacts_sba/
```
Variables: loan_amount, term, sba_guarantee_ratio, num_employees, created/retained
jobs, new_business, urban, real_estate_backed.

## Pool everything: cross-dataset attribution
Variables named consistently across converters (`fico`, `dti`, `int_rate`,
`loan_amount`, `term`) pool across datasets. `pool_impact.py` reports each
variable's effect per dataset AND pooled, and flags whether the *direction* is
consistent — a variable that moves default the same way across mortgage, consumer,
and small-business books is a robust signal to trust in the MCA model.
```bash
python research/pool_impact.py \
  --data export_freddie.json export_lc.json export_sba.json \
  --names freddie lending_club sba --out pooled_impact.json
```
Example (real data): FICO consistently protective and DTI consistently
risk-increasing across books → trust them; a variable with mixed signs → don't.

### How the harness uses each dataset
- **Time-series datasets** (Freddie, internal ledgers) → train + score the
  spectral challenger AND champion + attribution; produce a servable artifact.
- **Tabular-only datasets** (Lending Club, SBA) → champion + attribution only;
  no `weights.npz` (nothing for the spectral bridge to serve).

---


The model needs two things per deal: a **bank-ledger time-series** (features) and a
**realized outcome** (default/repaid label). Ranked by relevance.

## Tier 1 — internal, most relevant (you likely already have it)
PTM already stores merchants' **`bank_statement`** documents for funded deals
(`documents` table / `mca-docs` bucket) — that *is* the spectral input, on real
merchants. `applications` holds the funding terms (factor, term, payment,
funded amount). What's missing:
- **Labels** — `deal_outcomes` exists but is empty. Backfill it for past deals
  (defaulted? collected?). This is the single highest-value action.
- **Statement → ledger** — historical statements are PDFs; going forward the
  manual ledger entry (PTM lead profile) captures dated transactions with no API.

Export to the harness format with the SQL shape below, then `ingest_dataset.py`
or feed `export.json` directly. Real MCA data beats every public proxy.

## Tier 2 — public proxies (bootstrap + validate only, NOT for pricing)
No public dataset is exactly MCA (daily ACH remittance + factor outcomes). Best
matches, with honest caveats:

| Dataset | Has time-series? | Has default label? | Relevance / caveat |
| --- | --- | --- | --- |
| **Berka / PKDD'99 financial** | ✅ per-account transactions | ✅ loan good/bad | Best public match for the *spectral* method; Czech retail banking, not MCA. |
| **SBA 7(a) loans** (data.gov/Kaggle) | ❌ | ✅ charge-off | Small-business default + firmographics — good for the CFR *champion* baseline. |
| LendingClub / Prosper | ❌ | ✅ | Consumer installment loans; firmographic transfer only. |
| Freddie/Fannie performance | ✅ monthly | ✅ | Mortgages; useful only to pretrain temporal features. |
| PaySim / synthetic txns | ✅ | ⚠️ fraud, not default | Plumbing/scale tests; relabel needed. |

Licensing varies — review each before use; keep them out of the repo (size +
license). Use them to pretrain/validate the pipeline, then fine-tune on Tier 1.

## Where to download (verified)
- **Berka / PKDD'99 financial** — CTU Relational Dataset Repository:
  https://relational.fit.cvut.cz/dataset/Financial  (mirror:
  https://www.kaggle.com/datasets/marceloventura/the-berka-dataset). Tables:
  `account`, `trans` (1M+ transactions), `loan` (status A/C = good, B/D = bad).
  Best public match for the spectral method.
- **Home Credit Default Risk** — Kaggle competition (strong proxy: repayment
  time-series + binary default): https://www.kaggle.com/c/home-credit-default-risk/data
  Use `installments_payments.csv` (time-series) + `application_train.csv.TARGET`.
- **SBA 7(a) loans** — official FOIA: https://data.sba.gov/dataset/7-a-504-foia
  (also https://catalog.data.gov/dataset/sba-7a-and-504-loan-data-reports;
  Kaggle: https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set).
  Firmographics + charge-off label, NO ledger → champion/baseline only.
- LendingClub (Kaggle `wordsforthewise/lending-club`), Freddie Mac
  single-family performance (https://www.freddiemac.com/research/datasets,
  free registration), PaySim (`ealaxi/paysim1`) — lower relevance, see table.

### Per-dataset ingest mapping
```bash
# Berka: trans.csv has account_id, date, amount; build labels.csv from loan.csv
#   (status in {B,D} -> defaulted=1, else 0), then:
python research/ingest_dataset.py --transactions trans.csv --labels loan_labels.csv \
  --tx-account account_id --tx-date date --tx-amount amount \
  --lab-account account_id --lab-default defaulted --out export_berka.json

# Home Credit: installments_payments -> transactions (SK_ID_CURR, payment date, amount);
#   labels from application_train (SK_ID_CURR, TARGET):
python research/ingest_dataset.py --transactions installments.csv --labels app_train.csv \
  --tx-account SK_ID_CURR --tx-date DAYS_INSTALMENT --tx-amount AMT_PAYMENT \
  --lab-account SK_ID_CURR --lab-default TARGET --out export_homecredit.json
```

## Newer datasets for variable → default attribution
When the goal is tracking *how specific quantitative variables move default*
(not just a binary label), prefer recent sets with named, interpretable fields:

| Dataset | Recency | Named variables? | Time-series? | Best for |
| --- | --- | --- | --- | --- |
| **Freddie Mac / Fannie Mae single-family** | current (quarterly to present) | ✅ FICO, LTV, DTI, rate, UPB, loan age, delinquency | ✅ monthly | Transparent variable→default attribution on current data |
| **Amex Default Prediction** (2022) | recent | ⚠️ masked, grouped (D_/S_/P_/B_/R_) | ✅ monthly per customer | Newest match for the *spectral* method; category-level impact only |
| **Lending Club** (through 2018Q4) | mid | ✅ income, DTI, FICO, revolving util, grade | ❌ | Rich interpretable tabular attribution |
| **UCI "Default of Credit Card Clients"** | mid | ✅ limit, age, 6-mo bill/pay history | ⚠️ 6 months | Clean, small, fully interpretable |

Links: Freddie Mac https://www.freddiemac.com/research/datasets (free registration) ·
Amex https://www.kaggle.com/competitions/amex-default-prediction ·
Lending Club https://www.kaggle.com/datasets/wordsforthewise/lending-club ·
UCI https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Caveat: Freddie/Amex give the freshest data but neither is MCA; use for
attribution methodology + pretraining, validate on internal Tier-1 deals.

## Measuring variable impact (built in)
`variable_impact.py` runs automatically inside the harness and writes
`variable_impact.json`: per-variable **Information Value (IV)**, **default rate
per quantile bin**, and **point-biserial correlation** + the champion logistic's
signed weights. Works on any ingested dataset, so you can rank which quantitative
variables drive default and in which direction.

## One ingestion path for all of them
`ingest_dataset.py` maps any `transactions.csv` + `labels.csv` (configurable
column names) into `export.json`, so Berka, an SBA-style table, or a PTM export
all train through `train_calibrate_backtest.py`.

### Internal PTM export (SQL sketch)
```sql
-- transactions.csv
SELECT interested_party_id AS account_id, posted_at AS date, amount
FROM daily_ledger_sync ORDER BY interested_party_id, posted_at;

-- labels.csv (only matured deals — no look-ahead)
SELECT o.interested_party_id AS account_id,
       o.defaulted::int      AS defaulted,
       o.funded_at,
       p.monthly_revenue     AS monthly_deposits_avg,
       p.existing_positions  AS current_positions
FROM deal_outcomes o
JOIN interested_parties p ON p.id = o.interested_party_id
WHERE o.matured = true;
```

## Honest bottom line
Until Tier 1 labels accrue, public proxies can prove the *method* and pretrain
the spectral representation, but they cannot calibrate MCA pricing. The backtest
gate (`promote_challenger`) stays the guardrail: the quantum model ships only
when it beats the CFR champion on *your* matured deals.
