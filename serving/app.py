"""ASGI entrypoint for the underwriting bridge.

Run locally:   uvicorn app:app --host 0.0.0.0 --port 8000
In K3s:        see deploy/k8s/ (single-replica StatefulSet owns the weights).
"""
from fastapi import FastAPI

from underwriting_api import MODEL_VERSION, router

app = FastAPI(title="QuantumCortex Underwriting Bridge", version=MODEL_VERSION)
app.include_router(router)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}
