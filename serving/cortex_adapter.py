"""Adapter between the QuantumCortex engine and MCA underwriting.

Loads a versioned artifact produced by research/train_calibrate_backtest.py:
  weights.npz       -> trained W_in / W_lat
  calibration.json  -> num_classes + Platt (raw spectral risk -> calibrated PD)

The engine classifies risk bands; the *last* class is "default", so raw risk =
P(default) from the readout energies, then Platt scaling turns it into a
calibrated PD. PRICING IS PTM'S JOB: the bridge emits calibrated_pd +
spectral_risk_score; PTM's rateDeal() turns PD -> LGD -> EL -> grade -> terms so
there is one pricing authority. The suggested factor/holdback below exist only
for the batch cron path and are explicitly secondary.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourier_optics import FourierOptics  # noqa: E402
from quantum_cortex import QuantumCortex  # noqa: E402

DEFAULT_VERSION = "qc-uw-untrained"
NUM_INPUTS = 3136
NEURONS_PER_BAND = 15  # must match the training harness
INTERFERENCE_PD = 0.50  # calibrated PD at/above which we raise the default warning

# Suggested-pricing envelope (cron path only; PTM is the pricing authority).
FACTOR_MIN, FACTOR_MAX = 1.10, 1.49
HOLDBACK_MIN, HOLDBACK_MAX = 0.08, 0.25


class UnderwritingCortex:
    def __init__(self, snapshot_path: str | None = None, artifact_dir: str | None = None):
        self.optics = FourierOptics(shape=(28, 28))
        self.num_classes = 2
        self.platt = (1.0, 0.0)  # identity until calibrated
        self.model_version = DEFAULT_VERSION

        artifact_dir = artifact_dir or os.environ.get("CORTEX_ARTIFACT_DIR")
        if artifact_dir and os.path.isdir(artifact_dir):
            self._load_calibration(os.path.join(artifact_dir, "calibration.json"))

        self.cortex = QuantumCortex(NUM_INPUTS, self.num_classes, NEURONS_PER_BAND)

        # Trained weights: artifact takes precedence, else a live snapshot.
        if artifact_dir and os.path.exists(os.path.join(artifact_dir, "weights.npz")):
            self._load_weights(os.path.join(artifact_dir, "weights.npz"))
        elif snapshot_path and os.path.exists(snapshot_path):
            self._load_weights(snapshot_path)

        self.snapshot_path = snapshot_path

    # -- inference -----------------------------------------------------------
    def evaluate(self, image: np.ndarray) -> dict:
        risk = self._raw_risk(image)
        return self._result(risk)

    # -- online learning (O(1) Hebbian update) -------------------------------
    def learn(self, image: np.ndarray, realized_band: int) -> dict:
        features = self.optics.apply(image)
        self.cortex.process_image(features, label=realized_band, train=False)
        result = self._result(self._risk_from_energies(self.cortex.last_class_energies))
        self.cortex.process_image(features, realized_band, train=True)  # the update
        if self.snapshot_path:
            self.save(self.snapshot_path)
        return result

    # -- scoring -------------------------------------------------------------
    def _raw_risk(self, image: np.ndarray) -> float:
        features = self.optics.apply(image)
        self.cortex.process_image(features, label=0, train=False)
        return self._risk_from_energies(self.cortex.last_class_energies)

    def _risk_from_energies(self, class_energies: np.ndarray) -> float:
        e = np.asarray(class_energies, dtype=float)
        total = e.sum()
        if total <= 0:
            return 0.5
        return float(e[-1] / total)  # last class = default

    def _calibrated_pd(self, raw_risk: float) -> float:
        a, b = self.platt
        z = a * raw_risk + b
        pd = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
        return float(np.clip(pd, 0.0, 1.0))

    def _result(self, raw_risk: float) -> dict:
        pd = self._calibrated_pd(raw_risk)
        return {
            "calibrated_pd": round(pd, 4),
            "spectral_risk_score": round(pd, 4),
            "destructive_interference_flag": bool(pd >= INTERFERENCE_PD),
            "model_version": self.model_version,
            # Suggested only (cron path). PTM prices authoritatively via rateDeal.
            "calculated_factor_rate": round(FACTOR_MIN + (FACTOR_MAX - FACTOR_MIN) * pd, 4),
            "dynamic_holdback_percentage": round(
                float(np.clip(HOLDBACK_MIN + (HOLDBACK_MAX - HOLDBACK_MIN) * pd,
                              HOLDBACK_MIN, HOLDBACK_MAX)), 4),
        }

    # -- artifact / persistence ---------------------------------------------
    def _load_calibration(self, path: str) -> None:
        if not os.path.exists(path):
            return
        data = json.load(open(path))
        self.num_classes = int(data.get("num_classes", self.num_classes))
        p = data.get("platt", {})
        self.platt = (float(p.get("a", 1.0)), float(p.get("b", 0.0)))
        self.model_version = data.get("model_version", self.model_version)

    def _load_weights(self, path: str) -> None:
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        self.cortex.W_in = data["W_in"]
        self.cortex.W_lat = data["W_lat"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, W_in=self.cortex.W_in, W_lat=self.cortex.W_lat)
