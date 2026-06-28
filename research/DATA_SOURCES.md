# Training data for the underwriting model

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
