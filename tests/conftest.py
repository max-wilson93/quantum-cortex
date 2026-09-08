import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runtime  # noqa: E402,F401  # pins BLAS threads; must precede numpy

import data as data_module  # noqa: E402

# Small but real. Every test is seeded, so results are deterministic.
N_TRAIN = 400
N_TEST = 200
SEED = 0


@pytest.fixture(scope="session")
def split():
    try:
        return data_module.make_split(SEED, N_TRAIN, N_TEST)
    except FileNotFoundError as exc:
        pytest.skip(f"MNIST not available: {exc}")


@pytest.fixture(scope="session")
def trained_cortex(split):
    """One column trained on the fixture split. Shared: training is the slow part."""
    from quantum_cortex import QuantumCortex

    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=SEED)
    for i in range(split.n_train):
        cortex.process_image(split.features_train[i], split.labels_train[i], train=True)
    return cortex
