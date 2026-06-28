"""Adapter between the QuantumCortex engine and MCA underwriting terms.

The engine is a *classifier* over 3 risk bands (0=distress, 1=stable,
2=healthy), not a learned pricer. So `score()` turns per-band readout energies
into a calibrated risk in [0,1], and `terms()` maps that risk to a factor
rate / holdback. That mapping is a documented HEURISTIC — replace it with a
backtested calibration (or a regression readout head) before pricing real money.

The model is stateful: online Hebbian learning mutates W_in/W_lat in place, so
one process owns the authoritative weights and snapshots them to disk (mounted
to a PVC in K3s) so a restart doesn't lose what it has learned.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# The engine + optics live at the repo root, one level up from serving/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourier_optics import FourierOptics  # noqa: E402
from quantum_cortex import QuantumCortex  # noqa: E402

MODEL_VERSION = "qc-uw-0.1"
NUM_BANDS = 3           # distress(0) / stable(1) / healthy(2)
NEURONS_PER_BAND = 15
NUM_INPUTS = 3136       # 4 oriented masks * 28 * 28, matches FourierOptics

# Heuristic pricing envelope. Tune against realized repayment before production.
FACTOR_MIN, FACTOR_MAX = 1.10, 1.49
HOLDBACK_MIN, HOLDBACK_MAX = 0.08, 0.25
INTERFERENCE_RISK = 0.80  # distress band + risk over this => default warning


class UnderwritingCortex:
    def __init__(self, snapshot_path: str | None = None):
        self.optics = FourierOptics(shape=(28, 28))
        self.cortex = QuantumCortex(NUM_INPUTS, NUM_BANDS, NEURONS_PER_BAND)
        self.snapshot_path = snapshot_path
        if snapshot_path and os.path.exists(snapshot_path):
            self.load(snapshot_path)

    # -- inference -----------------------------------------------------------
    def evaluate(self, image: np.ndarray) -> dict:
        features = self.optics.apply(image)
        _, pred, _ = self.cortex.process_image(features, label=0, train=False)
        risk = self._risk_from_energies(self.cortex.last_class_energies)
        return {**self._terms(pred, risk), "predicted_band": int(pred)}

    # -- online learning (O(1) Hebbian update) -------------------------------
    def learn(self, image: np.ndarray, realized_band: int) -> dict:
        features = self.optics.apply(image)
        # Inference snapshot first (terms reflect state *before* this update)...
        _, pred, _ = self.cortex.process_image(features, label=realized_band, train=False)
        risk = self._risk_from_energies(self.cortex.last_class_energies)
        terms = {**self._terms(pred, risk), "predicted_band": int(pred)}
        # ...then the single Hebbian weight update on the realized outcome.
        self.cortex.process_image(features, realized_band, train=True)
        if self.snapshot_path:
            self.save(self.snapshot_path)
        return terms

    # -- mapping -------------------------------------------------------------
    @staticmethod
    def _risk_from_energies(class_energies: np.ndarray) -> float:
        e = np.asarray(class_energies, dtype=float)
        total = e.sum()
        if total <= 0:
            return 0.5
        p = e / total  # [P(distress), P(stable), P(healthy)]
        # Risk = mass on distress, minus mass on healthy, mapped to [0,1].
        return float(np.clip(0.5 + 0.5 * (p[0] - p[NUM_BANDS - 1]), 0.0, 1.0))

    @staticmethod
    def _terms(pred: int, risk: float) -> dict:
        factor = round(FACTOR_MIN + (FACTOR_MAX - FACTOR_MIN) * risk, 4)
        holdback = round(
            float(np.clip(HOLDBACK_MIN + (HOLDBACK_MAX - HOLDBACK_MIN) * risk,
                          HOLDBACK_MIN, HOLDBACK_MAX)),
            4,
        )
        return {
            "calculated_factor_rate": factor,
            "dynamic_holdback_percentage": holdback,
            "spectral_risk_score": round(risk, 4),
            "destructive_interference_flag": bool(pred == 0 and risk >= INTERFERENCE_RISK),
            "model_version": MODEL_VERSION,
        }

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, W_in=self.cortex.W_in, W_lat=self.cortex.W_lat)

    def load(self, path: str) -> None:
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        self.cortex.W_in = data["W_in"]
        self.cortex.W_lat = data["W_lat"]
