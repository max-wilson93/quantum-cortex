# QuantumCortex Underwriting Bridge

FastAPI service that connects PTM's Supabase data layer to the QuantumCortex
engine for MCA underwriting. Lives next to the engine because it vendors
`quantum_cortex.py` + `fourier_optics.py` and adds a Lomb-Scargle front-end
suited to irregularly-sampled bank ledgers.

> **Honest scope note.** The engine is a *classifier* over 3 risk bands, not a
> learned pricer. `cortex_adapter._terms()` maps risk → factor rate / holdback
> with a **documented heuristic envelope**. Calibrate it against realized
> repayment (or add a regression readout head) before pricing real money.
>
> This is NumPy, not PyTorch — there is no Torch anywhere in the engine. The
> deployment is K3s-ready (`deploy/k8s/`) without a framework rewrite.

## Components
| File | Role |
| --- | --- |
| `preprocessing.py` | Lomb-Scargle periodogram on irregular ledger → 28×28 image |
| `cortex_adapter.py` | persistent engine; risk scoring, term mapping, weight snapshots |
| `underwriting_api.py` | routes: `/evaluate-lead/{id}`, `/learn-lead/{id}`, `/learn-active` |
| `app.py` | ASGI app + `/healthz` |
| `Dockerfile` | container (build from repo root) |

## Endpoints
- `POST /api/v1/evaluate-lead/{lead_id}` — inference. Reads `daily_ledger_sync`,
  writes `calculated_factor_rate`, `dynamic_holdback_percentage`,
  `spectral_risk_score`, `destructive_interference_flag` back to
  `interested_parties`; inserts a `syndication_ratings` provenance row.
- `POST /api/v1/learn-lead/{lead_id}` — one online Hebbian update from fresh
  ledger data, then refresh the holdback.
- `POST /api/v1/learn-active` — batch sweep over funded deals (the daily CronJob).

All endpoints require the `x-ingest-secret` header (= PTM's `INGEST_SHARED_SECRET`).

## Continuous-learning loop (Phase 4 data flow)

```
Post-funding bank API sync (Plaid/MX webhook or nightly pull)
        │  upsert rows  (idempotent on provider + provider_txn_id)
        ▼
  daily_ledger_sync  ─────────────────────────────────────────────┐
        │                                                          │
   K3s CronJob (06:30 daily)  ── POST /api/v1/learn-active         │
        ▼                                                          │
  For each funded deal:                                            │
    1. ledger rows ─ Lomb-Scargle ─ 28×28 spectral image          │
    2. realized risk band  (from repayment/NSF signals)           │
    3. cortex.process_image(img, band, train=True)  ← O(1) Hebbian │
    4. re-read band energies → risk → new holdback                │
    5. UPDATE interested_parties.dynamic_holdback_percentage ──────┘
        ▼
  np.savez weights → PVC  (survives pod restart / reschedule)
```

**Why O(1):** the Hebbian update touches only the active input columns of the
target/wrong band blocks (phase-rotate + grow / damp) — no global gradient, no
backprop, constant work per sample regardless of history length.

**Why single-replica:** online learning is sequential; one pod owns the
authoritative weights (StatefulSet + PVC). Scale inference separately with a
read-only Deployment that loads published snapshots.

## Local run
```bash
pip install -r requirements.txt
export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... INGEST_SHARED_SECRET=...
export CORTEX_SNAPSHOT_PATH=./cortex_state.npz
uvicorn app:app --reload   # from serving/
```

## Deploy (K3s)
```bash
docker build -f serving/Dockerfile -t quantum-cortex-uw:0.1 .
kubectl create secret generic quantum-cortex-uw-secrets \
  --from-literal=SUPABASE_URL=... \
  --from-literal=SUPABASE_SERVICE_ROLE_KEY=... \
  --from-literal=INGEST_SHARED_SECRET=...
kubectl apply -f deploy/k8s/
```
