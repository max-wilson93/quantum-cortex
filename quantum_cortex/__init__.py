"""The Quantum-Holographic Cortex: online, one-shot, non-backpropagation.

A spiking network whose neurons are resonators rather than switches. Dendritic
integration is complex wave interference, non-linearity comes from the optical
Kerr effect, and a unitary energy clamp keeps a recurrent column stable. It
learns from a single presentation of each sample, with a local Hebbian rule on
phase and no global error gradient.

    >>> import numpy as np
    >>> from quantum_cortex import QuantumCortex, FourierOptics
    >>> optics = FourierOptics(shape=(28, 28))
    >>> cortex = QuantumCortex(optics.n_features, num_classes=10, seed=7)
    >>> features = optics.apply(np.zeros((28, 28)))
    >>> prediction = cortex.observe(features, label=3)
    >>> prediction.label, round(prediction.margin, 3)
    (0, 0.0)

Two properties are what make it worth wiring into an underwriting engine.
It learns **online** -- each outcome updates the model the moment it arrives,
rather than waiting for a retraining run. And it learns from **one** example,
which matters most exactly where conventional fitting refuses: a lender with
eleven logged decisions, or a bank layout seen once.

Two cautions travel with it. :attr:`~quantum_cortex.readout.Prediction.distribution`
is normalised energy and has never been calibrated against observed
frequencies -- it is not a probability, and turning it into one is a separate
evidenced step. And the three-cortex ensemble the architecture is named for was
measured to contribute nothing at its default settings; see
:mod:`quantum_cortex.ensemble` for the numbers.
"""

from quantum_cortex.cortex import GOLDEN_CONFIG, QuantumCortex
from quantum_cortex.encoders import (
    Encoder,
    FourierOptics,
    Passthrough,
    PhasicEncoding,
    RadialBands,
    SpectralSeries,
    TabularEncoder,
    to_phasic,
)
from quantum_cortex.ensemble import Ensemble
from quantum_cortex.readout import Prediction

__version__ = "0.2.0"

__all__ = [
    "GOLDEN_CONFIG",
    "Encoder",
    "Ensemble",
    "FourierOptics",
    "Passthrough",
    "PhasicEncoding",
    "Prediction",
    "QuantumCortex",
    "RadialBands",
    "SpectralSeries",
    "TabularEncoder",
    "to_phasic",
    "__version__",
]
