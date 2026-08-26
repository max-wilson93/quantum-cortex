"""The margin has to rank correctness, or no head can abstain.

All three planned underwriting heads need to be able to say "I don't know" --
a lender ranking with no confidence attached is a guess with a number on it.
The margin is the signal that makes that possible, so it gets its own file.

Measured on MNIST (6000 train / 2000 test): accuracy rises monotonically from
51.00% in the lowest margin decile to 99.50% in the highest, giving an AUC of
0.8641 against correctness. What follows is the same property on a small
synthetic problem that runs in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cortex import PhasicEncoding, QuantumCortex


def _separable_problem(
    rng: np.random.Generator, n: int, noise: float
) -> list[tuple[np.ndarray, int]]:
    """Two classes on opposite halves of the vector, blurred by ``noise``.

    At noise 0 the classes are trivially separable; as noise rises the cortex
    should not merely get more answers wrong, it should get *less confident*
    about the ones it gets wrong. That second property is the one under test.
    """
    left = np.concatenate([np.ones(8), np.zeros(8)])
    right = np.concatenate([np.zeros(8), np.ones(8)])

    samples = []
    for i in range(n):
        label = i % 2
        base = right if label else left
        sample = np.clip(base + rng.normal(0, noise, size=16), 0.0, 1.0)
        samples.append((sample, label))
    return samples


def _trained(noise: float, seed: int = 3) -> QuantumCortex:
    rng = np.random.default_rng(seed)
    cortex = QuantumCortex(16, 2, seed=seed, encoding=PhasicEncoding.MAGNITUDE)
    for features, label in _separable_problem(rng, 200, noise):
        cortex.observe(features, label)
    return cortex


class TestMarginRanksCorrectness:
    def test_correct_predictions_carry_higher_margins(self) -> None:
        """The core property. Without it there is no usable abstention."""
        cortex = _trained(noise=0.55)
        rng = np.random.default_rng(99)

        correct, wrong = [], []
        for features, label in _separable_problem(rng, 400, noise=0.55):
            prediction = cortex.predict(features)
            (correct if prediction.label == label else wrong).append(prediction.margin)

        assert wrong, "test needs some errors to compare against"
        assert np.mean(correct) > np.mean(wrong)

    def test_margin_separates_hits_from_misses_better_than_chance(self) -> None:
        """Mann-Whitney AUC of margin against correctness. 0.5 is no signal."""
        cortex = _trained(noise=0.55)
        rng = np.random.default_rng(101)

        margins, hits = [], []
        for features, label in _separable_problem(rng, 400, noise=0.55):
            prediction = cortex.predict(features)
            margins.append(prediction.margin)
            hits.append(prediction.label == label)

        margins_arr, hits_arr = np.array(margins), np.array(hits)
        good, bad = margins_arr[hits_arr], margins_arr[~hits_arr]
        assert len(bad) > 0

        wins = sum((good > b).sum() + 0.5 * (good == b).sum() for b in bad)
        auc = wins / (len(good) * len(bad))
        assert auc > 0.65, f"margin barely ranks correctness (AUC {auc:.3f})"

    def test_a_harder_problem_lowers_confidence(self) -> None:
        """Confidence should fall with signal, not just accuracy."""
        rng = np.random.default_rng(7)
        clean = _trained(noise=0.05)
        murky = _trained(noise=0.75)

        probe = _separable_problem(rng, 200, noise=0.4)
        clean_margin = np.mean([clean.predict(f).margin for f, _ in probe])
        murky_margin = np.mean([murky.predict(f).margin for f, _ in probe])
        assert clean_margin > murky_margin


class TestThresholdingIsRelative:
    """The margin's scale moves with training; only its ordering is stable."""

    def test_more_training_shifts_the_margin_scale(self) -> None:
        """Why `confident(min_margin=...)` must never take a hardcoded constant.

        On MNIST the median margin falls from 0.107 at 6k training samples to
        below 0.05 at 60k. A threshold tuned against the first abstains on
        essentially everything under the second.
        """
        rng = np.random.default_rng(5)
        probe = [f for f, _ in _separable_problem(rng, 100, noise=0.3)]

        def median_margin(n_samples: int) -> float:
            train_rng = np.random.default_rng(5)
            cortex = QuantumCortex(16, 2, seed=5, encoding=PhasicEncoding.MAGNITUDE)
            for features, label in _separable_problem(train_rng, n_samples, noise=0.3):
                cortex.observe(features, label)
            return float(np.median([cortex.predict(f).margin for f in probe]))

        assert median_margin(40) != pytest.approx(median_margin(600), rel=0.05)

    def test_a_quantile_threshold_holds_the_abstention_rate(self) -> None:
        """The supported pattern: take the cut from a holdout, store it."""
        cortex = _trained(noise=0.5)
        rng = np.random.default_rng(21)
        holdout = [cortex.predict(f).margin for f, _ in _separable_problem(rng, 300, 0.5)]

        cut = float(np.quantile(holdout, 0.2))
        live = [cortex.predict(f).margin for f, _ in _separable_problem(rng, 300, 0.5)]
        abstained = sum(1 for m in live if m < cut) / len(live)
        assert 0.1 < abstained < 0.35
