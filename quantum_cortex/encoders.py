"""Turning a thing in the world into something the cortex can resonate with.

Measurement drove this module's existence. Three ensemble configurations were
benchmarked on 6000 train / 2000 test MNIST samples:

===========================================  ========  ========  =============
configuration                                  single  ensemble  disagreements
===========================================  ========  ========  =============
identical init (as shipped)                    88.25%    88.25%        0/2000
random phase init (as the README describes)    88.05%    88.20%       14/2000
three radial frequency bands, one per lobe     58.90%    57.00%     1569/2000
===========================================  ========  ========  =============

The shipped "Trinity" contributes nothing -- three cortices that never once
disagreed. Random phase initialisation buys 14 disagreements and no accuracy,
because the phase-Hebbian rule rotates every active weight toward zero and so
walks all three cortices into the same attractor whatever they started from.
Splitting the spectrum three ways produces genuine diversity and three learners
too weak to vote usefully.

What carried all three runs was the same thing: the concatenated four-orientation
Fourier front end. **The encoder is the model's actual contribution**, and it is
therefore the extension point rather than the ensemble.

Two axes, kept separate because they answer different questions:

``Encoder``
    what features to extract from a sample -- spectral bands from an image,
    frequency content from a cash-flow series, normalised columns from a table.

:class:`PhasicEncoding`
    how a feature vector becomes a complex wave. The original binarised at a
    threshold, which is defensible for MNIST pixels and destroys a financial
    feature: a merchant at 0.71 and one at 6.0 both become exactly ``1+0j``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.fft as fft

__all__ = [
    "Encoder",
    "PhasicEncoding",
    "to_phasic",
    "FourierOptics",
    "RadialBands",
    "Passthrough",
    "TabularEncoder",
    "SpectralSeries",
]


class PhasicEncoding(StrEnum):
    """How a real-valued feature vector becomes a complex input wave."""

    BINARY = "binary"
    """Magnitude 1 above the threshold, 0 below; phase always 0.

    The original behaviour, and the one the 90.74% MNIST result was measured
    with. Correct for near-binary inputs like thresholded pixels. It throws
    away every distinction above the threshold, so it is the wrong choice for
    continuous features.
    """

    MAGNITUDE = "magnitude"
    """Magnitude is the (clipped) feature value; phase always 0.

    Preserves ordering and relative size. The plain choice for continuous
    features when their scale is already meaningful.
    """

    PHASE = "phase"
    """Unit magnitude; the feature value rides in the phase, over [0, 2pi).

    The encoding most native to the architecture -- this is a machine that
    integrates by interference, and phase is what interferes. Two merchants
    differing slightly in one feature produce waves that are slightly out of
    step rather than differing in brightness.
    """

    GATED_PHASE = "gated_phase"
    """Magnitude gated at the threshold, value in the phase.

    Both at once: the gate removes noise below the threshold, and everything
    that survives keeps its magnitude distinction in phase. The default for
    tabular financial features.
    """


def to_phasic(
    features: np.ndarray,
    *,
    encoding: PhasicEncoding = PhasicEncoding.BINARY,
    threshold: float = 0.7,
) -> np.ndarray:
    """Turn a real feature vector into a complex input wave.

    ``features`` is expected on roughly [0, 1]; :class:`TabularEncoder` is what
    puts arbitrary financial quantities onto that scale. Values outside it are
    clipped rather than rejected, because a merchant slightly past the top of
    the training range should saturate, not raise.
    """
    flat = np.asarray(features, dtype=float).ravel()

    match encoding:
        case PhasicEncoding.BINARY:
            magnitude = np.where(flat > threshold, 1.0, 0.0)
            phase = np.zeros_like(flat)
        case PhasicEncoding.MAGNITUDE:
            magnitude = np.clip(flat, 0.0, 1.0)
            phase = np.zeros_like(flat)
        case PhasicEncoding.PHASE:
            magnitude = np.ones_like(flat)
            phase = np.clip(flat, 0.0, 1.0) * 2.0 * np.pi
        case PhasicEncoding.GATED_PHASE:
            magnitude = np.where(flat > threshold, 1.0, 0.0)
            phase = np.clip(flat, 0.0, 1.0) * 2.0 * np.pi
        case _:  # pragma: no cover - StrEnum is exhaustive
            raise ValueError(f"unknown phasic encoding {encoding!r}")

    return magnitude * np.exp(1j * phase)


@runtime_checkable
class Encoder(Protocol):
    """Extracts a real-valued feature vector from one sample.

    Deliberately narrow. An encoder does not know what a class is, does not
    hold learned state, and is called identically during training and scoring
    -- so an encoder can never be the route by which a label leaks into
    inference.
    """

    @property
    def n_features(self) -> int:
        """Length of the vector :meth:`apply` returns.

        Needed before any sample exists, because it sizes the cortex.
        """

    def apply(self, sample: np.ndarray) -> np.ndarray:
        """Encode one sample into a feature vector on roughly [0, 1]."""


class Passthrough:
    """Uses the sample as its own feature vector.

    For features that arrived already prepared, and for tests that want the
    cortex isolated from any encoding at all.
    """

    def __init__(self, n_features: int) -> None:
        self._n = int(n_features)

    @property
    def n_features(self) -> int:
        return self._n

    def apply(self, sample: np.ndarray) -> np.ndarray:
        flat = np.asarray(sample, dtype=float).ravel()
        if flat.size != self._n:
            raise ValueError(f"expected {self._n} features, got {flat.size}")
        return flat


class FourierOptics:
    """Four oriented spectral bands, concatenated. The validated front end.

    A 4f optical correlator: transform to the spectral domain, mask to an
    orientation and a radial band, transform back. Four orientations at 45
    degrees apart, each band-passed to drop DC and the highest frequencies,
    simulating V1 receptive fields.

    This is the encoder every headline result in this repository was measured
    with, and per the module docstring it is doing most of the work. Left
    numerically identical to the original implementation.
    """

    def __init__(
        self,
        shape: tuple[int, int] = (28, 28),
        *,
        orientations: int = 4,
        bandwidth: float = np.pi / 8,
        radius_band: tuple[float, float] = (1.0, 14.0),
    ) -> None:
        self.rows, self.cols = shape
        self.radius_band = radius_band
        crow, ccol = self.rows // 2, self.cols // 2
        y, x = np.ogrid[-crow:self.rows - crow, -ccol:self.cols - ccol]
        theta = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)
        inner, outer = radius_band

        self.masks: list[np.ndarray] = []
        for k in range(orientations):
            target = k * np.pi / orientations
            diff = np.abs(theta - target)
            diff = np.minimum(diff, 2 * np.pi - diff)
            self.masks.append((diff < bandwidth) & (radius > inner) & (radius < outer))

    @property
    def n_features(self) -> int:
        return len(self.masks) * self.rows * self.cols

    def apply(self, sample: np.ndarray) -> np.ndarray:
        image = np.asarray(sample, dtype=float).reshape(self.rows, self.cols)
        spectrum = fft.fftshift(fft.fft2(image))
        bands = []
        for mask in self.masks:
            spatial = fft.ifft2(fft.ifftshift(spectrum * mask))
            magnitude = np.abs(spatial)
            peak = np.max(magnitude)
            if peak > 0:
                magnitude = magnitude / peak
            bands.append(magnitude.flatten())
        return np.concatenate(bands)


class RadialBands:
    """Splits the spectrum by radius: structure, shape, texture.

    Kept because it is the right decomposition for a *signal* -- the cash-flow
    work in :class:`SpectralSeries` is the two-dimensional case of the same
    idea -- and because it is the honest record of a measurement. Feeding one
    band to each of three cortices scored 57.00% against 88.25% for the
    concatenated orientation bands. Use it concatenated, or not at all.
    """

    def __init__(
        self,
        shape: tuple[int, int] = (28, 28),
        bands: tuple[tuple[float, float], ...] = ((1, 5), (5, 10), (10, 14)),
    ) -> None:
        self.rows, self.cols = shape
        crow, ccol = self.rows // 2, self.cols // 2
        y, x = np.ogrid[-crow:self.rows - crow, -ccol:self.cols - ccol]
        radius = np.sqrt(x**2 + y**2)
        self.masks = [(radius > lo) & (radius <= hi) for lo, hi in bands]

    @property
    def n_features(self) -> int:
        return len(self.masks) * self.rows * self.cols

    def apply(self, sample: np.ndarray) -> np.ndarray:
        image = np.asarray(sample, dtype=float).reshape(self.rows, self.cols)
        spectrum = fft.fftshift(fft.fft2(image))
        bands = []
        for mask in self.masks:
            spatial = fft.ifft2(fft.ifftshift(spectrum * mask))
            magnitude = np.abs(spatial)
            peak = np.max(magnitude)
            if peak > 0:
                magnitude = magnitude / peak
            bands.append(magnitude.flatten())
        return np.concatenate(bands)


class TabularEncoder:
    """Financial features onto [0, 1], by rank against a reference sample.

    The encoder the underwriting heads need, and the one the original code had
    no equivalent of. Three properties matter and none of them is optional:

    **Rank, not scale.** Underwriting features are heavily right-skewed --
    monthly revenue, position count, NSF counts. Min-max scaling lets one
    merchant at $900k/month compress every other merchant into the bottom
    tenth of the range. Mapping each feature to its quantile against a fitted
    reference sample is scale-free and outlier-proof, and it is the same
    device PTM's lender-fit envelopes already use for the same reason.

    **A missing feature is not a zero.** A merchant with no FICO on file is not
    a merchant with a bad FICO. ``NaN`` encodes to :attr:`missing_value`, whose
    default sits below any sane gate threshold, so an unmeasured feature
    contributes no energy rather than contributing wrong energy.

    **Direction is explicit.** ``lower_is_better`` flips a feature's rank, so
    that a high encoded value always means "more of the thing the cortex should
    resonate with". Leaving direction implicit is how a scorecard ends up
    inverted, which the underwriting engine has been bitten by before.
    """

    def __init__(
        self,
        feature_names: list[str],
        *,
        lower_is_better: frozenset[str] = frozenset(),
        quantiles: int = 64,
        missing_value: float = 0.0,
    ) -> None:
        if not feature_names:
            raise ValueError("a tabular encoder needs at least one feature")
        self.feature_names = list(feature_names)
        self.lower_is_better = frozenset(lower_is_better)
        self.quantiles = int(quantiles)
        self.missing_value = float(missing_value)
        self._knots: np.ndarray | None = None

        unknown = self.lower_is_better - set(self.feature_names)
        if unknown:
            raise ValueError(f"lower_is_better names unknown features: {sorted(unknown)}")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def fitted(self) -> bool:
        return self._knots is not None

    def fit(self, samples: np.ndarray) -> TabularEncoder:
        """Learn the quantile knots from a reference sample.

        ``samples`` is ``(n_samples, n_features)``. ``NaN`` is ignored per
        column rather than per row, so a column that is missing for some
        merchants still gets knots from the merchants that have it.
        """
        matrix = np.asarray(samples, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(
                f"expected (n_samples, {self.n_features}), got {matrix.shape}"
            )
        levels = np.linspace(0.0, 1.0, self.quantiles)
        knots = np.empty((self.n_features, self.quantiles), dtype=float)
        for col in range(self.n_features):
            values = matrix[:, col]
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                # Nothing was ever measured. Flat knots encode everything to
                # the same place, which is the truthful outcome.
                knots[col, :] = 0.0
            else:
                knots[col, :] = np.quantile(observed, levels)
        self._knots = knots
        return self

    def apply(self, sample: np.ndarray) -> np.ndarray:
        if self._knots is None:
            raise RuntimeError("TabularEncoder.fit must be called before apply")
        values = np.asarray(sample, dtype=float).ravel()
        if values.size != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {values.size}")

        out = np.empty(self.n_features, dtype=float)
        for col, value in enumerate(values):
            if not np.isfinite(value):
                out[col] = self.missing_value
                continue
            knots = self._knots[col]
            if knots[-1] == knots[0]:
                out[col] = 0.5
                continue
            rank = float(np.searchsorted(knots, value, side="right")) / len(knots)
            rank = min(max(rank, 0.0), 1.0)
            if self.feature_names[col] in self.lower_is_better:
                rank = 1.0 - rank
            out[col] = rank
        return out


class SpectralSeries:
    """A one-dimensional series into normalised frequency bands.

    Built for a merchant's daily balance or remittance stream. A cash-flow
    series has real periodic structure -- weekly payroll, monthly rent, the
    near-constant daily pull of an advance -- and separating those bands is
    exactly what an FFT front end is for. The underwriting engine already
    reaches for autocorrelation and an FFT spectral peak on a banking-day index
    to discriminate an MCA pull from ordinary recurring ACH, so this is the
    same signal read the same way.

    The series is detrended before transforming. Without that, the level term
    dominates every band and the encoder reports how large the business is
    rather than how its cash behaves.
    """

    def __init__(self, length: int, *, bands: int = 8, detrend: bool = True) -> None:
        if length < 4:
            raise ValueError("a spectral encoder needs at least 4 observations")
        if bands < 1:
            raise ValueError("bands must be positive")
        self.length = int(length)
        self.bands = int(bands)
        self.detrend = detrend
        usable = self.length // 2
        self._edges = np.linspace(1, usable, self.bands + 1).astype(int)

    @property
    def n_features(self) -> int:
        return self.bands

    def apply(self, sample: np.ndarray) -> np.ndarray:
        series = np.asarray(sample, dtype=float).ravel()
        if series.size != self.length:
            raise ValueError(f"expected {self.length} observations, got {series.size}")
        if not np.all(np.isfinite(series)):
            raise ValueError("series carries non-finite values; resample before encoding")

        if self.detrend:
            index = np.arange(series.size, dtype=float)
            slope, intercept = np.polyfit(index, series, 1)
            series = series - (slope * index + intercept)

        power = np.abs(fft.rfft(series)) ** 2
        out = np.empty(self.bands, dtype=float)
        for k in range(self.bands):
            lo, hi = self._edges[k], self._edges[k + 1]
            segment = power[lo:hi] if hi > lo else power[lo:lo + 1]
            out[k] = float(np.sum(segment))

        total = float(np.sum(out))
        if total <= 0.0:
            # A perfectly flat series after detrending. No band dominates, and
            # saying so beats dividing by zero.
            return np.full(self.bands, 1.0 / self.bands)
        return out / total
