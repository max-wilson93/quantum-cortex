"""Phase 0 sped the update up. This proves it did not change it.

`quantum_cortex.QuantumCortex` replaced a per-neuron Python loop and a
full-matrix clip with vectorised, touched-entries-only equivalents. That is a
7-8x speedup, which is what makes 5-seed benchmarking affordable -- but a
speedup that quietly alters the model would poison every number in results.md.

`tests/reference_cortex.py` is a frozen transcription of the original loop.
Both are driven from the same seeded generator and must agree.
"""

import numpy as np
import pytest

from quantum_cortex import QuantumCortex
from reference_cortex import ReferenceCortex

N_SAMPLES = 400


@pytest.fixture(scope="module")
def pair(split):
    fast = QuantumCortex(split.num_features, split.num_classes, 5, seed=7)
    slow = ReferenceCortex(split.num_features, split.num_classes, 5, seed=7)
    fast_predictions, slow_predictions = [], []
    for i in range(min(N_SAMPLES, split.n_train)):
        features, label = split.features_train[i], split.labels_train[i]
        fast_predictions.append(fast.process_image(features, label, train=True)[1])
        slow_predictions.append(slow.process_image(features, label, train=True)[1])
    return fast, slow, np.array(fast_predictions), np.array(slow_predictions)


def test_training_predictions_are_identical(pair):
    _, _, fast_predictions, slow_predictions = pair
    assert np.array_equal(fast_predictions, slow_predictions)


def test_weights_agree_to_round_off(pair):
    fast, slow, _, _ = pair
    # The reference re-clips and re-composes every weight on every sample, so
    # untouched entries accumulate a phase/magnitude round-trip the fast path
    # skips. The residual is that round-off, not a difference in the rule.
    assert np.max(np.abs(fast.W_in - slow.W_in)) < 1e-10
    assert np.array_equal(fast.W_lat, slow.W_lat)


def test_evaluation_predictions_are_identical(pair, split):
    fast, slow, _, _ = pair
    for i in range(split.n_test):
        features, label = split.features_test[i], split.labels_test[i]
        assert (fast.process_image(features, label, train=False)[1]
                == slow.process_image(features, label, train=False)[1])
