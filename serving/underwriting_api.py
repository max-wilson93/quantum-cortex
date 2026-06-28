"""FastAPI bridge: Supabase (PTM) <-> QuantumCortex underwriting engine.

Endpoints
  POST /api/v1/evaluate-lead/{lead_id}  -> inference; writes terms back to PTM.
  POST /api/v1/learn-lead/{lead_id}     -> online Hebbian update from fresh ledger
                                           data; updates the dynamic holdback.

Auth reuses PTM's existing INGEST_SHARED_SECRET header convention so no new
secret model is introduced. DB access uses the Supabase service-role key, like
PTM's cron routes.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException
from supabase import Client, create_client

from cortex_adapter import MODEL_VERSION, UnderwritingCortex
from preprocessing import InsufficientLedgerError, ledger_to_spectral_image

router = APIRouter(prefix="/api/v1")

_SECRET = os.environ["INGEST_SHARED_SECRET"]
_SNAPSHOT = os.environ.get("CORTEX_SNAPSHOT_PATH", "/data/cortex_state.npz")
_ARTIFACT_DIR = os.environ.get("CORTEX_ARTIFACT_DIR")  # trained weights + calibration

_sb: Client = create_client(
    os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)
# Single in-process model (authoritative weights; see cortex_adapter docstring).
_model = UnderwritingCortex(snapshot_path=_SNAPSHOT, artifact_dir=_ARTIFACT_DIR)


def _auth(secret: str) -> None:
    if secret != _SECRET:
        raise HTTPException(status_code=401, detail="bad ingest secret")


def _load_ledger(lead_id: str) -> list[dict]:
    res = (
        _sb.table("daily_ledger_sync")
        .select("posted_at,amount,txn_type,running_balance")
        .eq("interested_party_id", lead_id)
        .order("posted_at")
        .execute()
    )
    return res.data or []


def _write_terms(lead_id: str, terms: dict) -> None:
    _sb.table("interested_parties").update(
        {
            "calculated_factor_rate": terms["calculated_factor_rate"],
            "dynamic_holdback_percentage": terms["dynamic_holdback_percentage"],
            "spectral_risk_score": terms["spectral_risk_score"],
            "destructive_interference_flag": terms["destructive_interference_flag"],
            "uw_model_version": terms["model_version"],
            "uw_evaluated_at": "now()",
        }
    ).eq("id", lead_id).execute()


@router.post("/evaluate-lead/{lead_id}")
def evaluate_lead(
    lead_id: str,
    persist: bool = True,
    x_ingest_secret: str = Header(default=""),
):
    """Score a lead. With persist=false, only computes + returns (no DB writes) —
    used by PTM's interactive action, which then prices + persists authoritatively
    via its own economics."""
    _auth(x_ingest_secret)
    rows = _load_ledger(lead_id)
    try:
        image = ledger_to_spectral_image(rows)
    except InsufficientLedgerError as e:
        raise HTTPException(status_code=422, detail=str(e))

    terms = _model.evaluate(image)
    if not persist:
        return {"lead_id": lead_id, **terms}

    _write_terms(lead_id, terms)
    # Record a syndication rating row capturing the spectral provenance.
    _sb.table("syndication_ratings").insert(
        {
            "interested_party_id": lead_id,
            "factor_rate": terms["calculated_factor_rate"],
            "pd": terms["calibrated_pd"],
            "spectral_risk_score": terms["spectral_risk_score"],
            "uw_model_version": terms["model_version"],
        }
    ).execute()
    return {"lead_id": lead_id, **terms}


@router.post("/learn-lead/{lead_id}")
def learn_lead(lead_id: str, x_ingest_secret: str = Header(default="")):
    """Continuous-learning step: re-score on fresh ledger data and apply one
    O(1) Hebbian update toward the realized risk band, then refresh the holdback.
    """
    _auth(x_ingest_secret)
    rows = _load_ledger(lead_id)
    try:
        image = ledger_to_spectral_image(rows)
    except InsufficientLedgerError as e:
        raise HTTPException(status_code=422, detail=str(e))

    realized_band = _realized_band(rows)
    terms = _model.learn(image, realized_band)
    _write_terms(lead_id, terms)
    return {"lead_id": lead_id, "realized_band": realized_band, **terms}


@router.post("/learn-active")
def learn_active(x_ingest_secret: str = Header(default="")):
    """Batch continuous-learning sweep, driven by the daily K3s CronJob.

    Pulls every funded deal and applies one Hebbian update + holdback refresh
    each. Funded deals are read from PTM (stage = 'funded'); their fresh ledger
    rows arrive via the daily banking sync into daily_ledger_sync.
    """
    _auth(x_ingest_secret)
    funded = (
        _sb.table("interested_parties")
        .select("id")
        .eq("stage", "funded")
        .execute()
        .data
        or []
    )
    updated, skipped = 0, 0
    for row in funded:
        lead_id = row["id"]
        try:
            image = ledger_to_spectral_image(_load_ledger(lead_id))
        except InsufficientLedgerError:
            skipped += 1
            continue
        terms = _model.learn(image, _realized_band(_load_ledger(lead_id)))
        _write_terms(lead_id, terms)
        updated += 1
    return {"funded": len(funded), "updated": updated, "skipped": skipped}


def _realized_band(rows: list[dict]) -> int:
    """Derive the realized risk band from recent repayment behavior.

    PROXY: we don't have a remittance feed wired yet, so we read distress signals
    from the ledger tail (NSFs, negative running balance). Swap this for actual
    remittance on-time/missed data once that table exists.
        0 = distress, 1 = stable, 2 = healthy
    """
    tail = rows[-20:]
    nsf = sum(1 for r in tail if (r.get("txn_type") or "").lower() == "nsf")
    negs = sum(
        1
        for r in tail
        if r.get("running_balance") is not None and float(r["running_balance"]) < 0
    )
    if nsf >= 2 or negs >= 5:
        return 0
    if nsf == 0 and negs == 0:
        return 2
    return 1


__all__ = ["router", "MODEL_VERSION"]
