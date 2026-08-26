"""What the cortex reports when it has finished resonating.

The original readout returned ``argmax`` over class energies and nothing else.
That is enough to score MNIST and not enough to underwrite anything: a credit
decision needs to know *how strongly* the model believes, and needs to be able
to say it does not know. A label alone cannot express either.

``Prediction`` therefore carries the whole energy vector, a normalised
distribution over it, and a margin. The margin is the honest confidence signal:
the gap between the winning class and the runner-up, as a share of total
energy. A cortex that splits its energy evenly between two classes reports a
margin near zero regardless of which one happens to win the ``argmax``, and a
caller can refuse to act on that.

The distribution is deliberately **not** called a probability. It is normalised
energy, it has never been calibrated against observed frequencies, and naming
it a probability is exactly the error that would let it be read as one. Mapping
it to a real probability is a separate, evidenced step -- in the underwriting
engine that is what the calibration ladder is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Prediction"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """One cortex's answer, with enough context to decide whether to trust it."""

    label: int
    """The winning class. ``argmax`` over :attr:`energies`."""

    energies: np.ndarray
    """Raw energy per class. Length ``num_classes``."""

    total_energy: float
    """Summed energy across every output neuron, before class pooling.

    Low total energy means the input barely excited the cortex at all, which is
    a different failure from energy that is high but evenly spread.
    """

    @property
    def distribution(self) -> np.ndarray:
        """Energies normalised to sum to 1.

        Normalised energy, **not** a probability -- see the module docstring.
        Returns a uniform distribution when no class carried any energy, which
        is the honest answer to "the input excited nothing".
        """
        total = float(np.sum(self.energies))
        if total <= 0.0:
            return np.full(len(self.energies), 1.0 / len(self.energies))
        return np.asarray(self.energies / total, dtype=float)

    @property
    def margin(self) -> float:
        """Gap between the winner and the runner-up, as a share of total energy.

        ``0.0`` when the top two classes are tied, ``1.0`` when one class holds
        every unit of energy. This is the quantity to threshold on when the
        caller needs an abstention.
        """
        if len(self.energies) < 2:
            return 1.0
        ranked = np.sort(self.distribution)[::-1]
        return float(ranked[0] - ranked[1])

    @property
    def runner_up(self) -> int:
        """The second-choice class. What the cortex would have said instead."""
        if len(self.energies) < 2:
            return self.label
        return int(np.argsort(self.energies)[::-1][1])

    def confident(self, *, min_margin: float) -> bool:
        """Whether the margin clears ``min_margin``.

        Kept as a method taking an explicit threshold rather than a property
        with a default, because the right threshold is a property of the
        decision being made and there is no defensible universal value.
        """
        return self.margin >= min_margin

    def ranked(self) -> list[tuple[int, float]]:
        """Every class with its normalised energy, best first.

        The lender-acceptance head needs a ranking, not a winner -- "who is
        most likely to fund this" is a list.
        """
        dist = self.distribution
        order = np.argsort(dist)[::-1]
        return [(int(c), float(dist[c])) for c in order]
