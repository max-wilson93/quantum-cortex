"""The ensemble, including the measurement that says it earns little.

These tests pin the finding rather than only the mechanics. If a future change
makes the default ensemble genuinely diverse, `test_identical_members_never_
disagree` fails and someone reads this file to find out why that is good news.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cortex import Ensemble, QuantumCortex
from quantum_cortex.readout import Prediction


def _members(n: int = 3, *, phase_init: str = "zero") -> list[QuantumCortex]:
    return [
        QuantumCortex(16, 3, seed=seed, phase_init=phase_init)  # type: ignore[arg-type]
        for seed in range(n)
    ]


def _stream(n: int = 90) -> list[tuple[np.ndarray, int]]:
    rng = np.random.default_rng(0)
    return [(rng.uniform(0, 1, size=16), int(i % 3)) for i in range(n)]


class TestConstruction:
    def test_needs_a_member(self) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            Ensemble([])

    def test_members_must_agree_on_the_class_count(self) -> None:
        mixed = [QuantumCortex(16, 3, seed=1), QuantumCortex(16, 4, seed=2)]
        with pytest.raises(ValueError, match="disagree on num_classes"):
            Ensemble(mixed)

    def test_unknown_consensus_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown consensus"):
            Ensemble(_members(), consensus="vibes")

    def test_bag_fraction_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="bag_fraction"):
            Ensemble(_members(), bag_fraction=0.0)


class TestTheMeasuredFinding:
    """As shipped, the Trinity contributes nothing. Pin it."""

    def test_identical_members_never_disagree(self) -> None:
        """No randomness in __init__ means three copies of one model.

        Measured on MNIST: 0 disagreements across 2000 held-out samples.
        """
        ensemble = Ensemble([QuantumCortex(16, 3, seed=7) for _ in range(3)])
        for features, label in _stream(30):
            ensemble.observe(features, label)
        rng = np.random.default_rng(1)
        for _ in range(30):
            assert ensemble.disagreement(rng.uniform(0, 1, size=16)) == 0.0

    def test_bagging_produces_members_that_actually_differ(self) -> None:
        """Different data is diversity the phase rule cannot anneal away."""
        ensemble = Ensemble(_members(), bag_fraction=0.4, seed=5)
        for features, label in _stream(120):
            ensemble.observe(features, label)
        assert ensemble.members[0].samples_seen != ensemble.members[1].samples_seen
        assert not np.array_equal(ensemble.members[0].W_in, ensemble.members[1].W_in)


class TestConsensus:
    def test_energy_consensus_keeps_confidence(self) -> None:
        """A confident minority can outweigh two lukewarm members."""
        readings = [
            Prediction(label=0, energies=np.array([0.51, 0.49]), total_energy=1.0),
            Prediction(label=0, energies=np.array([0.51, 0.49]), total_energy=1.0),
            Prediction(label=1, energies=np.array([0.02, 0.98]), total_energy=1.0),
        ]
        ensemble = Ensemble(_members(), consensus="energy")
        assert ensemble._combine(readings).label == 1

    def test_vote_consensus_discards_confidence(self) -> None:
        """Same readings, opposite answer. This is the trade-off, stated."""
        readings = [
            Prediction(label=0, energies=np.array([0.51, 0.49]), total_energy=1.0),
            Prediction(label=0, energies=np.array([0.51, 0.49]), total_energy=1.0),
            Prediction(label=1, energies=np.array([0.02, 0.98]), total_energy=1.0),
        ]
        ensemble = Ensemble(_members(), consensus="vote")
        assert ensemble._combine(readings).label == 0

    def test_energy_is_the_default(self) -> None:
        assert Ensemble(_members()).consensus == "energy"


class TestBehaviour:
    def test_predict_does_not_learn(self) -> None:
        ensemble = Ensemble(_members())
        for _ in range(5):
            ensemble.predict(np.ones(16))
        assert all(m.samples_seen == 0 for m in ensemble.members)

    def test_observe_learns_in_every_member_at_full_bag(self) -> None:
        ensemble = Ensemble(_members(), bag_fraction=1.0)
        for features, label in _stream(20):
            ensemble.observe(features, label)
        assert all(m.samples_seen == 20 for m in ensemble.members)

    def test_member_predictions_expose_the_disagreement(self) -> None:
        ensemble = Ensemble(_members())
        readings = ensemble.member_predictions(np.ones(16))
        assert len(readings) == len(ensemble)

    def test_decay_reaches_every_member(self) -> None:
        ensemble = Ensemble(_members())
        before = [m.learning_rate for m in ensemble.members]
        ensemble.decay_learning_rate(1.0)
        assert all(
            m.learning_rate < b
            for m, b in zip(ensemble.members, before, strict=True)
        )

    def test_trinity_uses_random_phases_and_distinct_seeds(self) -> None:
        ensemble = Ensemble.trinity(16, 3, seeds=(1, 2, 3))
        assert len(ensemble) == 3
        assert not np.array_equal(ensemble.members[0].W_in, ensemble.members[1].W_in)


class TestRoundTrip:
    def test_ensemble_survives_save_and_load(self, tmp_path) -> None:
        ensemble = Ensemble(_members(), consensus="energy")
        for features, label in _stream(30):
            ensemble.observe(features, label)

        ensemble.save(tmp_path / "book")
        restored = Ensemble.load(tmp_path / "book")

        assert len(restored) == len(ensemble)
        probe = np.random.default_rng(3).uniform(0, 1, size=16)
        assert restored.predict(probe).label == ensemble.predict(probe).label

    def test_loading_an_empty_directory_is_an_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="no ensemble members"):
            Ensemble.load(tmp_path)
