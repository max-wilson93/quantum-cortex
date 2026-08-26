"""The core: shape, determinism, learning, and the readout contract."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cortex import GOLDEN_CONFIG, PhasicEncoding, Prediction, QuantumCortex


def _sample(rng: np.random.Generator, n: int = 16) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=n)


class TestShape:
    """The original pooled ``range(10)`` whatever num_classes said."""

    @pytest.mark.parametrize("num_classes", [2, 3, 7, 10, 41])
    def test_readout_pools_into_num_classes(self, num_classes: int) -> None:
        cortex = QuantumCortex(16, num_classes, neurons_per_class=3, seed=1)
        prediction = cortex.predict(np.ones(16))
        assert len(prediction.energies) == num_classes
        assert 0 <= prediction.label < num_classes

    def test_a_binary_problem_is_expressible(self) -> None:
        """Default risk is two classes. Under the old readout it was ten."""
        cortex = QuantumCortex(8, num_classes=2, seed=1)
        assert cortex.num_outputs == 2 * cortex.neurons_per_class
        assert len(cortex.predict(np.ones(8)).energies) == 2

    def test_rejects_degenerate_geometry(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            QuantumCortex(16, num_classes=1, seed=1)
        with pytest.raises(ValueError, match="num_inputs"):
            QuantumCortex(0, num_classes=2, seed=1)

    def test_label_outside_the_class_range_is_refused(self) -> None:
        cortex = QuantumCortex(8, num_classes=3, seed=1)
        with pytest.raises(ValueError, match="outside"):
            cortex.observe(np.ones(8), label=3)


class TestLateralStrength:
    """It was accepted from the config and never read."""

    def test_lateral_strength_reaches_the_weights(self) -> None:
        loose = QuantumCortex(8, 3, config={**GOLDEN_CONFIG, "lateral_strength": 0.4}, seed=1)
        tight = QuantumCortex(8, 3, config={**GOLDEN_CONFIG, "lateral_strength": 0.05}, seed=1)
        assert np.abs(loose.W_lat).max() > np.abs(tight.W_lat).max()

    def test_the_documented_golden_value_is_what_the_code_ran(self) -> None:
        """The historical config records 0.16; the code applied a hardcoded 0.1.

        GOLDEN_CONFIG records the value that was actually in effect, because
        that is the one the published accuracy was measured with.
        """
        assert GOLDEN_CONFIG["lateral_strength"] == 0.10


class TestDeterminism:
    """The consuming engine guarantees byte-identical repeat runs."""

    def test_same_seed_same_weights(self) -> None:
        rng = np.random.default_rng(0)
        samples = [(_sample(rng), int(i % 3)) for i in range(60)]

        def train() -> QuantumCortex:
            cortex = QuantumCortex(16, 3, seed=99)
            for features, label in samples:
                cortex.observe(features, label)
            return cortex

        assert np.array_equal(train().W_in, train().W_in)
        assert np.array_equal(train().W_lat, train().W_lat)

    def test_different_seeds_diverge(self) -> None:
        """Only through the damping branch, which fires on error."""
        rng = np.random.default_rng(0)
        samples = [(_sample(rng), int(i % 3)) for i in range(60)]

        def train(seed: int) -> QuantumCortex:
            cortex = QuantumCortex(16, 3, seed=seed)
            for features, label in samples:
                cortex.observe(features, label)
            return cortex

        assert not np.array_equal(train(1).W_in, train(2).W_in)

    def test_prediction_does_not_mutate_the_model(self) -> None:
        cortex = QuantumCortex(16, 3, seed=5)
        before = cortex.W_in.copy()
        for _ in range(10):
            cortex.predict(np.ones(16))
        assert np.array_equal(cortex.W_in, before)
        assert cortex.samples_seen == 0


class TestLearning:
    def test_one_shot_separates_two_distinguishable_classes(self) -> None:
        """The whole claim: one presentation each, no epochs."""
        left = np.concatenate([np.ones(8), np.zeros(8)])
        right = np.concatenate([np.zeros(8), np.ones(8)])
        cortex = QuantumCortex(16, num_classes=2, seed=3)

        for _ in range(5):
            cortex.observe(left, 0)
            cortex.observe(right, 1)

        assert cortex.predict(left).label == 0
        assert cortex.predict(right).label == 1

    def test_observe_reports_belief_before_the_update(self) -> None:
        """Scoring after the update would be scoring against the answer."""
        features = np.ones(16)
        cortex = QuantumCortex(16, 3, seed=3)
        first = cortex.observe(features, label=2)
        assert first.label != 2 or cortex.samples_seen == 1
        assert cortex.samples_seen == 1

    def test_class_counts_track_what_was_seen(self) -> None:
        cortex = QuantumCortex(8, 3, seed=1)
        for _ in range(4):
            cortex.observe(np.ones(8), 1)
        cortex.observe(np.ones(8), 2)
        assert cortex.class_counts() == {0: 0, 1: 4, 2: 1}

    def test_annealing_lowers_plasticity(self) -> None:
        cortex = QuantumCortex(8, 3, seed=1)
        start = cortex.learning_rate
        cortex.decay_learning_rate(1.0)
        assert cortex.learning_rate < start

    def test_diagonal_of_lateral_coupling_stays_zero(self) -> None:
        """Self-coupling would let one neuron drive itself into resonance."""
        cortex = QuantumCortex(8, 3, seed=1)
        for i in range(20):
            cortex.observe(np.ones(8), i % 3)
        assert np.allclose(np.diag(cortex.W_lat), 0.0)

    def test_weights_stay_bounded_under_sustained_growth(self) -> None:
        cortex = QuantumCortex(8, 2, seed=1)
        for _ in range(200):
            cortex.observe(np.ones(8), 0)
        assert np.abs(cortex.W_in).max() <= 1.0 + 1e-12
        assert np.abs(cortex.W_lat).max() <= 0.5 + 1e-12
        assert np.all(np.isfinite(cortex.W_in))


class TestClassBalance:
    def test_rare_class_gets_more_plasticity(self) -> None:
        """A 12% default rate otherwise teaches the model to say 'no default'."""
        cortex = QuantumCortex(8, 2, seed=1, balance_classes=True)
        for _ in range(50):
            cortex.observe(np.ones(8), 0)
        cortex.observe(np.ones(8), 1)
        assert cortex._plasticity_for(1) > cortex._plasticity_for(0)

    def test_balancing_is_off_by_default(self) -> None:
        """It changes the validated numerics, so it is opt-in."""
        cortex = QuantumCortex(8, 2, seed=1)
        for _ in range(50):
            cortex.observe(np.ones(8), 0)
        assert cortex._plasticity_for(1) == cortex._plasticity_for(0)

    def test_scaling_is_clipped(self) -> None:
        cortex = QuantumCortex(8, 3, seed=1, balance_classes=True)
        for _ in range(500):
            cortex.observe(np.ones(8), 0)
        assert cortex._plasticity_for(2) <= cortex.learning_rate * 4.0


class TestReadout:
    def test_distribution_sums_to_one(self) -> None:
        prediction = QuantumCortex(8, 4, seed=1).predict(np.ones(8))
        assert prediction.distribution.sum() == pytest.approx(1.0)

    def test_margin_is_zero_when_energy_is_evenly_split(self) -> None:
        prediction = Prediction(label=0, energies=np.array([1.0, 1.0]), total_energy=2.0)
        assert prediction.margin == pytest.approx(0.0)

    def test_margin_is_one_when_a_class_takes_everything(self) -> None:
        prediction = Prediction(label=0, energies=np.array([1.0, 0.0]), total_energy=1.0)
        assert prediction.margin == pytest.approx(1.0)

    def test_dead_input_gives_a_uniform_distribution_not_a_crash(self) -> None:
        prediction = Prediction(label=0, energies=np.zeros(4), total_energy=0.0)
        assert prediction.distribution.tolist() == [0.25] * 4
        assert prediction.margin == pytest.approx(0.0)

    def test_confidence_thresholds_on_the_margin(self) -> None:
        prediction = Prediction(label=0, energies=np.array([0.8, 0.2]), total_energy=1.0)
        assert prediction.confident(min_margin=0.5)
        assert not prediction.confident(min_margin=0.7)

    def test_ranked_is_ordered_best_first(self) -> None:
        """The lender head needs a ranking, not a winner."""
        prediction = Prediction(label=1, energies=np.array([0.2, 0.5, 0.3]), total_energy=1.0)
        ranked = prediction.ranked()
        assert [c for c, _ in ranked] == [1, 2, 0]
        assert ranked[0][1] == pytest.approx(0.5)

    def test_runner_up_is_the_second_choice(self) -> None:
        prediction = Prediction(label=1, energies=np.array([0.2, 0.5, 0.3]), total_energy=1.0)
        assert prediction.runner_up == 2


class TestCompatibility:
    def test_the_original_call_shape_still_works(self) -> None:
        """main.py is the MNIST regression guard and calls this."""
        cortex = QuantumCortex(16, 10, 5, config=GOLDEN_CONFIG, seed=1)
        correct, predicted, energy = cortex.process_image(np.ones(16), 4, train=True)
        assert isinstance(correct, bool)
        assert 0 <= predicted < 10
        assert energy >= 0.0

    def test_train_false_does_not_learn(self) -> None:
        cortex = QuantumCortex(16, 10, seed=1)
        cortex.process_image(np.ones(16), 4, train=False)
        assert cortex.samples_seen == 0


class TestEncodingReachesTheCortex:
    def test_binary_gate_collapses_magnitude(self) -> None:
        """Why the default is wrong for financial features."""
        cortex = QuantumCortex(4, 2, seed=1, encoding=PhasicEncoding.BINARY)
        low = cortex.get_phasic_input(np.array([0.71, 0.71, 0.71, 0.71]))
        high = cortex.get_phasic_input(np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.array_equal(low, high)

    def test_gated_phase_keeps_them_apart(self) -> None:
        cortex = QuantumCortex(4, 2, seed=1, encoding=PhasicEncoding.GATED_PHASE)
        low = cortex.get_phasic_input(np.array([0.71, 0.71, 0.71, 0.71]))
        high = cortex.get_phasic_input(np.array([1.0, 1.0, 1.0, 1.0]))
        assert not np.array_equal(low, high)
