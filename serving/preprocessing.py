"""Spectral preprocessing for MCA bank-ledger underwriting.

Bank ledgers are *irregularly* sampled — deposits/withdrawals land on uneven
days. The QuantumCortex front-end (fourier_optics / scipy spectrogram) assumes
uniform spacing, so feeding raw ledger events through an FFT would alias. The
Lomb-Scargle periodogram is designed exactly for unevenly-sampled series, so we
use it to build the spectral "image" the cortex consumes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle
from skimage.transform import resize

MIN_TXNS = 16  # below this there isn't enough signal for a meaningful periodogram


class InsufficientLedgerError(ValueError):
    """Raised when a lead has too little ledger history to evaluate."""


def ledger_to_spectral_image(rows: list[dict]) -> np.ndarray:
    """Irregular ledger rows -> normalized 28x28 image for FourierOptics.apply().

    Each row needs at least ``posted_at`` (ISO str or datetime) and ``amount``.
    """
    if rows is None or len(rows) < MIN_TXNS:
        raise InsufficientLedgerError(
            f"need >= {MIN_TXNS} ledger rows, got {0 if not rows else len(rows)}"
        )

    rows = sorted(rows, key=lambda r: r["posted_at"])
    t = np.array(
        [np.datetime64(r["posted_at"]).astype("datetime64[D]").astype(float) for r in rows]
    )
    t = t - t.min()  # days since first transaction
    y = np.array([float(r["amount"]) for r in rows], dtype=float)

    # Lomb-Scargle requires a zero-mean signal and strictly increasing samples.
    y = y - y.mean()
    t, y = _dedupe_times(t, y)
    if len(t) < MIN_TXNS or np.allclose(y, 0.0):
        raise InsufficientLedgerError("degenerate ledger after cleaning")

    span = max(t.max(), 1.0)
    # Angular frequencies from ~the full span down to a 2-day cycle (Nyquist-ish
    # for daily cash movements). 256 bins -> resized to the optics grid.
    freqs = np.linspace(2 * np.pi / span, 2 * np.pi / 2.0, 256)
    power = lombscargle(t, y, freqs, normalize=True)

    img = np.tile(power, (16, 1))  # 1-D spectrum -> 2-D so the optics stage has structure
    img = resize(img, (28, 28), mode="reflect", anti_aliasing=True)
    rng = float(img.max() - img.min())
    return (img - img.min()) / rng if rng else np.zeros((28, 28))


def _dedupe_times(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse same-day events (sum amounts) so sample times are unique."""
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    uniq, idx = np.unique(t, return_inverse=True)
    summed = np.zeros_like(uniq, dtype=float)
    np.add.at(summed, idx, y)
    return uniq, summed
