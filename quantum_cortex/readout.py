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
    def relative(self) -> np.ndarray:
        """Energies as a share of the strongest class, each independent of the rest.

        The reading to use when classes are **not** mutually exclusive -- one
        merchant file can draw an offer from several lenders at once, and
        :attr:`distribution` would force those into competition by making them
        sum to 1, so a file every lender wants looks identical to a file only
        one lender wants.

        Each entry is that class's energy over the maximum, so the winner is
        always 1.0 and the others say how close behind they came.
        """
        peak = float(np.max(self.energies)) if len(self.energies) else 0.0
        if peak <= 0.0:
            return np.zeros(len(self.energies), dtype=float)
        return np.asarray(self.energies / peak, dtype=float)

    @property
    def runner_up(self) -> int:
        """The second-choice class. What the cortex would have said instead."""
        if len(self.energies) < 2:
            return self.label
        return int(np.argsort(self.energies)[::-1][1])

    def confident(self, *, min_margin: float) -> bool:
        """Whether the margin clears ``min_margin``.

        **Set the threshold from a quantile of a holdout, never as a constant.**
        The margin's ordering is a strong signal -- measured on MNIST, accuracy
        rises monotonically from 51% in the lowest margin decile to 99.5% in the
        highest, an AUC of 0.86 against correctness. Its *scale* is not stable:
        the same physics trained on 6,000 samples has a median margin of 0.107,
        and on 60,000 the median falls below 0.05, because more training drives
        more weights toward the clip bound and spreads the energy thinner.

        So a threshold that abstains sensibly for one model silently abstains on
        everything for the next. Take the quantile you want on held-out data and
        store it beside the model, which is also what makes the abstention rate
        auditable rather than incidental.

        A method taking an explicit threshold rather than a property with a
        default, for the same reason: there is no defensible universal value.
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
