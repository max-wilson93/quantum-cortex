"""Several cortices reading the same input, and an honest account of the gain.

The architecture describes three independently initialised cortical columns
whose outputs combine by constructive interference: a real topological signal
makes all three resonate in step, while noise leaves them out of phase and
cancels. That is a good idea. It is not, as shipped, what happens.

Measured on 6000 train / 2000 test MNIST samples:

===========================================  ========  ========  =============
configuration                                  single  ensemble  disagreements
===========================================  ========  ========  =============
identical init (as shipped)                    88.25%    88.25%        0/2000
random phase init                              88.05%    88.20%       14/2000
three radial bands, one per member             58.90%    57.00%     1569/2000
===========================================  ========  ========  =============

Two things go wrong, and they are different problems.

**As shipped there is no diversity to aggregate.** ``QuantumCortex.__init__``
contains no randomness, so three members are the same model three times. They
never disagreed once across 2000 samples; the vote was decorative and the
compute was tripled for it.

**Adding diversity at initialisation does not survive training.** The
phase-Hebbian rule rotates every active weight's phase *toward zero*. Whatever
phases a member starts from, the rule walks it into the same attractor as every
other member. Random initialisation bought 14 disagreements out of 2000 and no
measurable accuracy.

So an ensemble here needs its members to differ in something training cannot
anneal away. The two routes this module supports are different *data*
(``bag_fraction`` -- each member learns from its own subsample, so their
histories genuinely differ) and different *encoders* (pass distinct encoders
per member). The third route, splitting one encoder's bands across members, was
measured and is a trap: it produces plenty of disagreement and members too weak
for the disagreement to be worth resolving.

None of this makes an ensemble useless. It makes it something to justify with a
measurement on your own data rather than to switch on by default -- which is
why :func:`Ensemble.trinity` exists but nothing calls it for you.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from quantum_cortex.cortex import QuantumCortex
from quantum_cortex.readout import Prediction

__all__ = ["Ensemble", "Consensus"]

Consensus = str
"""How members' answers combine. ``"energy"`` or ``"vote"``."""


class Ensemble:
    """A set of cortices scored together.

    Args:
        members: The cortices. All must agree on ``num_classes``.
        consensus: ``"energy"`` sums each member's class energies, which is the
            constructive-interference reading the architecture describes and
            keeps every member's confidence in the result. ``"vote"`` counts
            argmax winners, discarding confidence, and cannot break a tie
            without an arbitrary rule. Energy is the default for both reasons.
        bag_fraction: Share of samples each member learns from during
            :meth:`observe`, drawn per member per sample. ``1.0`` means every
            member sees everything, which -- per the module docstring -- is the
            configuration measured to produce no diversity at all. Below 1.0
            the members' training histories genuinely differ.
        seed: Seeds the bagging draw. Member seeds are their own.
    """

    def __init__(
        self,
        members: Sequence[QuantumCortex],
        *,
        consensus: Consensus = "energy",
        bag_fraction: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        classes = {m.num_classes for m in members}
        if len(classes) > 1:
            raise ValueError(f"members disagree on num_classes: {sorted(classes)}")
        if consensus not in ("energy", "vote"):
            raise ValueError(f"unknown consensus {consensus!r}")
        if not 0.0 < bag_fraction <= 1.0:
            raise ValueError("bag_fraction must be in (0, 1]")

        self.members = list(members)
        self.consensus = consensus
        self.bag_fraction = float(bag_fraction)
        self.num_classes = self.members[0].num_classes
        self._rng = np.random.default_rng(seed)

    @classmethod
    def trinity(
        cls,
        num_inputs: int,
        num_classes: int,
        neurons_per_class: int = 5,
        *,
        config: dict[str, float] | None = None,
        seeds: tuple[int, ...] = (11, 22, 33),
        bag_fraction: float = 1.0,
        **kwargs: object,
    ) -> Ensemble:
        """Three cortices with distinct seeds and random phase initialisation.

        The architecture's headline configuration. Measured as near-inert at
        ``bag_fraction=1.0`` -- see the module docstring -- so if you reach for
        this, set a bag fraction below 1 and verify the gain on your own data
        before paying three times the compute for it.
        """
        members = [
            QuantumCortex(
                num_inputs,
                num_classes,
                neurons_per_class,
                config=config,
                seed=seed,
                phase_init="random",
                **kwargs,  # type: ignore[arg-type]
            )
            for seed in seeds
        ]
        return cls(members, bag_fraction=bag_fraction, seed=seeds[0])

    # -- scoring -----------------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> Prediction:
        """Score without learning, combining members by :attr:`consensus`."""
        return self._combine([m.predict(feature_vector) for m in self.members])

    def observe(self, feature_vector: np.ndarray, label: int) -> Prediction:
        """Score and learn. Members below the bag fraction sit this sample out.

        A member that skips a sample still contributes its (unlearned) reading
        to the returned consensus, because the caller asked what the ensemble
        believed before the update, and half an ensemble is not that.
        """
        readings: list[Prediction] = []
        for member in self.members:
            learns = self.bag_fraction >= 1.0 or self._rng.random() < self.bag_fraction
            if learns:
                readings.append(member.observe(feature_vector, label))
            else:
                readings.append(member.predict(feature_vector))
        return self._combine(readings)

    def member_predictions(self, feature_vector: np.ndarray) -> list[Prediction]:
        """Each member's own answer, unaggregated.

        The diagnostic that tells you whether an ensemble is earning its keep:
        if these never differ, it is not.
        """
        return [m.predict(feature_vector) for m in self.members]

    def disagreement(self, feature_vector: np.ndarray) -> float:
        """Share of members that differ from the ensemble's own answer.

        ``0.0`` means unanimity. Sustained zero across a held-out set means the
        members are the same model and the ensemble is pure overhead.
        """
        readings = self.member_predictions(feature_vector)
        consensus = self._combine(readings).label
        differing = sum(1 for r in readings if r.label != consensus)
        return differing / len(readings)

    def _combine(self, readings: Sequence[Prediction]) -> Prediction:
        if self.consensus == "energy":
            # Constructive interference: members that agree reinforce, members
            # that disagree spread their energy and cancel.
            pooled = np.sum([r.distribution for r in readings], axis=0)
        else:
            pooled = np.zeros(self.num_classes, dtype=float)
            for reading in readings:
                pooled[reading.label] += 1.0
        return Prediction(
            label=int(np.argmax(pooled)),
            energies=pooled,
            total_energy=float(np.sum([r.total_energy for r in readings])),
        )

    # -- schedule and persistence -----------------------------------------

    def decay_learning_rate(self, progress: float) -> None:
        """Anneal every member's plasticity together."""
        for member in self.members:
            member.decay_learning_rate(progress)

    def save(self, directory: Path | str) -> list[Path]:
        """Write each member to ``member_000.npz`` and so on."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        return [
            member.save(target / f"member_{i:03d}.npz")
            for i, member in enumerate(self.members)
        ]

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        consensus: Consensus = "energy",
        bag_fraction: float = 1.0,
        seed: int | None = None,
    ) -> Ensemble:
        """Reconstruct an ensemble written by :meth:`save`."""
        source = Path(directory)
        files = sorted(source.glob("member_*.npz"))
        if not files:
            raise FileNotFoundError(f"no ensemble members found in {source}")
        members = [QuantumCortex.load(path) for path in files]
        return cls(members, consensus=consensus, bag_fraction=bag_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self.members)
