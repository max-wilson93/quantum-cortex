"""Several independent outcomes on one sample, and weighted evidence.

Lender acceptance is not a single-label problem. PTM's `lender_offers` indexes
(lenderFileId, lenderId) non-uniquely, so one merchant file is shopped to a
handful of lenders and each answers for itself: three offers and two declines
is five observations about one file, not one label.

The property that carries the most weight here is the third state. A lender the
file was never submitted to has not declined it, and training it as a negative
would teach the cortex your submission habits rather than the lenders' credit
boxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cortex import PhasicEncoding, QuantumCortex


def _cortex(classes: int = 6, seed: int = 3) -> QuantumCortex:
    return QuantumCortex(12, classes, seed=seed, encoding=PhasicEncoding.MAGNITUDE)


class TestUnobservedIsNotNegative:
    """The reason observe_multi exists rather than a positives-only signature."""

    def test_a_class_in_neither_set_is_untouched(self) -> None:
        cortex = _cortex()
        untouched = cortex._columns_for(5)
        before = cortex.W_in[:, untouched].copy()

        cortex.observe_multi(np.ones(12), positives=[0, 1], negatives=[2, 3])

        assert np.array_equal(cortex.W_in[:, untouched], before)

    def test_positives_grow_and_negatives_shrink(self) -> None:
        cortex = _cortex()
        features = np.ones(12)
        accepted, declined = cortex._columns_for(0), cortex._columns_for(2)
        before_accept = np.abs(cortex.W_in[:, accepted]).sum()
        before_decline = np.abs(cortex.W_in[:, declined]).sum()

        cortex.observe_multi(features, positives=[0], negatives=[2])

        assert np.abs(cortex.W_in[:, accepted]).sum() > before_accept
        assert np.abs(cortex.W_in[:, declined]).sum() < before_decline

    def test_never_submitted_lenders_do_not_drift_toward_decline(self) -> None:
        """The failure this prevents: learning who you ask, not who says yes."""
        cortex = _cortex(classes=6)
        rng = np.random.default_rng(1)
        never_asked = cortex._columns_for(5)
        before = cortex.W_in[:, never_asked].copy()

        for _ in range(50):
            features = rng.uniform(0, 1, size=12)
            cortex.observe_multi(features, positives=[0, 1], negatives=[2, 3, 4])

        assert np.array_equal(cortex.W_in[:, never_asked], before)


class TestMultiLabelLearning:
    def test_it_learns_which_lenders_take_which_files(self) -> None:
        """Two file archetypes, two disjoint sets of lenders that fund them."""
        strong = np.concatenate([np.ones(6), np.zeros(6)])
        weak = np.concatenate([np.zeros(6), np.ones(6)])
        cortex = _cortex(classes=4, seed=11)

        for _ in range(12):
            # Lenders 0 and 1 take strong files; 2 and 3 take weak ones.
            cortex.observe_multi(strong, positives=[0, 1], negatives=[2, 3])
            cortex.observe_multi(weak, positives=[2, 3], negatives=[0, 1])

        strong_ranked = [c for c, _ in cortex.predict(strong).ranked()]
        weak_ranked = [c for c, _ in cortex.predict(weak).ranked()]

        assert set(strong_ranked[:2]) == {0, 1}
        assert set(weak_ranked[:2]) == {2, 3}

    def test_every_positive_is_counted(self) -> None:
        cortex = _cortex(classes=4)
        cortex.observe_multi(np.ones(12), positives=[0, 2], negatives=[1])
        assert cortex.class_counts() == {0: 1, 1: 0, 2: 1, 3: 0}
        assert cortex.samples_seen == 1

    def test_declines_alone_are_a_valid_observation(self) -> None:
        """A file every lender turned down is real, and worth learning from."""
        cortex = _cortex()
        cortex.observe_multi(np.ones(12), positives=[], negatives=[0, 1, 2])
        assert cortex.samples_seen == 1


class TestMultiLabelValidation:
    def test_a_class_cannot_be_both(self) -> None:
        cortex = _cortex()
        with pytest.raises(ValueError, match="both positive and negative"):
            cortex.observe_multi(np.ones(12), positives=[1], negatives=[1])

    def test_an_out_of_range_class_is_refused(self) -> None:
        cortex = _cortex(classes=4)
        with pytest.raises(ValueError, match="outside"):
            cortex.observe_multi(np.ones(12), positives=[9])

    def test_an_empty_observation_is_refused(self) -> None:
        """Nothing observed is not the same as nothing happened."""
        cortex = _cortex()
        with pytest.raises(ValueError, match="at least one observed outcome"):
            cortex.observe_multi(np.ones(12), positives=[], negatives=[])

    def test_duplicates_collapse(self) -> None:
        cortex = _cortex()
        cortex.observe_multi(np.ones(12), positives=[1, 1, 1])
        assert cortex.class_counts()[1] == 1


class TestRelativeReadout:
    """distribution forces exclusivity; relative does not."""

    def test_relative_peaks_at_one(self) -> None:
        prediction = _cortex().predict(np.ones(12))
        assert prediction.relative.max() == pytest.approx(1.0)

    def test_relative_does_not_make_lenders_compete(self) -> None:
        """Under distribution, a file everyone wants looks like a file one wants."""
        everyone = QuantumCortex(12, 4, seed=1).predict(np.ones(12))
        # Two files with the same shape but different absolute strength give
        # identical distributions, while relative preserves the per-class read.
        assert everyone.distribution.sum() == pytest.approx(1.0)
        assert everyone.relative.sum() >= 1.0

    def test_dead_input_gives_zeros_not_nan(self) -> None:
        from quantum_cortex.readout import Prediction

        prediction = Prediction(label=0, energies=np.zeros(3), total_energy=0.0)
        assert prediction.relative.tolist() == [0.0, 0.0, 0.0]


class TestSampleWeight:
    """Partial observation is weaker evidence, and must push proportionally less."""

    def test_a_lighter_sample_moves_the_weights_less(self) -> None:
        def moved(weight: float) -> float:
            cortex = _cortex(seed=7)
            before = cortex.W_in.copy()
            cortex.observe(np.ones(12), 1, weight=weight)
            return float(np.abs(cortex.W_in - before).sum())

        assert moved(0.25) < moved(1.0) < moved(2.0)

    def test_zero_weight_moves_neither_magnitude_nor_phase(self) -> None:
        """Both halves, deliberately.

        An earlier version scaled only the learning rate, so a zero-weight
        sample left magnitudes alone and still rotated every active weight's
        phase at full strength. In a phase-Hebbian rule that is most of the
        learning, so the weight was only half-applied.
        """
        cortex = _cortex(seed=7)
        before = cortex.W_in.copy()
        cortex.observe(np.ones(12), 1, weight=0.0)

        assert np.allclose(np.abs(cortex.W_in), np.abs(before))
        assert np.allclose(np.angle(cortex.W_in), np.angle(before))
        assert np.array_equal(cortex.W_in, before)

    def test_a_zero_weight_sample_is_still_counted(self) -> None:
        """It was seen. The audit trail should say so even at no influence."""
        cortex = _cortex(seed=7)
        cortex.observe(np.ones(12), 1, weight=0.0)
        assert cortex.samples_seen == 1

    def test_weight_is_clipped(self) -> None:
        """Bad arithmetic upstream should cost one sample, not the model."""
        cortex = _cortex(seed=7)
        cortex.observe(np.ones(12), 1, weight=1e9)
        assert np.all(np.isfinite(cortex.W_in))
        assert np.abs(cortex.W_in).max() <= 1.0 + 1e-12

    def test_weight_reaches_the_multi_label_path(self) -> None:
        def moved(weight: float) -> float:
            cortex = _cortex(seed=7)
            before = cortex.W_in.copy()
            cortex.observe_multi(np.ones(12), positives=[0], negatives=[1], weight=weight)
            return float(np.abs(cortex.W_in - before).sum())

        assert moved(0.2) < moved(1.0)

    def test_censoring_weight_is_expressible(self) -> None:
        """The Head C pattern: observed_days / term_days.

        Movement is measured against the starting weights, not as a total
        magnitude -- growth on the positive and damping on the negative cancel
        in the sum, so |W_in|.sum() is very nearly invariant and would report
        no difference however the weight was applied.
        """

        def moved(weight: float) -> float:
            cortex = _cortex(classes=2, seed=5)
            before = cortex.W_in.copy()
            cortex.observe(np.ones(12), 1, weight=weight)
            return float(np.abs(cortex.W_in - before).sum())

        watched_20_of_90 = moved(20 / 90)
        watched_to_maturity = moved(1.0)
        assert watched_20_of_90 < watched_to_maturity


class TestSingleLabelIsUnchanged:
    """observe() must stay bit-identical: the MNIST result depends on it."""

    def test_default_weight_leaves_the_rule_alone(self) -> None:
        rng = np.random.default_rng(0)
        samples = [(rng.uniform(0, 1, size=12), int(i % 3)) for i in range(40)]

        def train(**kwargs: float) -> QuantumCortex:
            cortex = QuantumCortex(12, 3, seed=42)
            for features, label in samples:
                cortex.observe(features, label, **kwargs)  # type: ignore[arg-type]
            return cortex

        assert np.array_equal(train().W_in, train(weight=1.0).W_in)

    def test_single_label_damps_only_the_wrong_winner(self) -> None:
        """Not every non-target class -- that is the multi-label rule."""
        cortex = QuantumCortex(12, 4, seed=1)
        prediction = cortex.predict(np.ones(12))
        untouched = next(
            c for c in range(4) if c not in (prediction.label, (prediction.label + 1) % 4)
        )
        before = cortex.W_in[:, cortex._columns_for(untouched)].copy()

        cortex.observe(np.ones(12), (prediction.label + 1) % 4)

        assert np.array_equal(cortex.W_in[:, cortex._columns_for(untouched)], before)
