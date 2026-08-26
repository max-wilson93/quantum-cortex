"""Does the three-cortex ensemble do anything? Measured, not assumed.

The architecture is named for a three-column "Quantum Trinity" whose outputs
combine by constructive interference, described as quantum error correction: a
real signal makes all three resonate in step, noise leaves them out of phase and
cancels.

This script tests that claim three ways and prints the numbers. It is kept as
runnable code rather than a paragraph in a README because the conclusion drove a
design decision -- the ensemble is opt-in and the encoder is the extension point
-- and anyone who wants to overturn that decision should be able to re-measure
in one command rather than take it on trust.

    python benchmarks/ensemble_diversity.py            # 6000/2000, ~4 minutes
    python benchmarks/ensemble_diversity.py --full     # the whole corpus

Findings at 6000 train / 2000 test, seed 12345:

    identical init (as shipped)   88.25% single  88.25% ensemble     0/2000 disagree
    random phase init             88.05% single  88.20% ensemble    14/2000 disagree
    three radial bands            58.90% best    57.00% ensemble  1569/2000 disagree

The first is inert: ``QuantumCortex.__init__`` has no randomness, so three
members are one model three times. The second stays inert because the
phase-Hebbian rule rotates active weights toward zero phase and walks every
member into the same attractor regardless of where it started. The third
produces real diversity and members too weak for it to be worth resolving --
the ensemble scores below its own best member.

What carried every run was the concatenated four-orientation Fourier front end.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from quantum_cortex import Ensemble, FourierOptics, QuantumCortex, RadialBands
from quantum_cortex.cortex import GOLDEN_CONFIG
from quantum_cortex.datasets import LocalMNISTLoader

DATA_PATH = "./mnist_data"
NEURONS_PER_CLASS = 5
SHUFFLE_SEED = 12345


@dataclass(frozen=True, slots=True)
class Result:
    name: str
    best_single: float
    ensemble: float
    disagreements: int
    rescued: int
    samples: int
    seconds: float

    def report(self) -> str:
        return (
            f"{self.name:<34} "
            f"single {self.best_single:6.2f}%  "
            f"ensemble {self.ensemble:6.2f}%  "
            f"disagree {self.disagreements:5d}/{self.samples}  "
            f"rescued {self.rescued:4d}  "
            f"({self.seconds:.0f}s)"
        )


def load(train_n: int, test_n: int):
    loader = LocalMNISTLoader(DATA_PATH)
    train_x = loader.load_images("train-images.idx3-ubyte")
    train_y = loader.load_labels("train-labels.idx1-ubyte")
    test_x = loader.load_images("t10k-images.idx3-ubyte")
    test_y = loader.load_labels("t10k-labels.idx1-ubyte")

    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(train_x))[:train_n]
    return train_x[order], train_y[order], test_x[:test_n], test_y[:test_n]


def measure(name, members, encode, data) -> Result:
    """Train each member on its own view of the sample, then score the vote."""
    train_x, train_y, test_x, test_y = data
    start = time.time()

    for i in range(len(train_x)):
        views = encode(train_x[i])
        label = int(train_y[i])
        for member, view in zip(members, views, strict=True):
            member.observe(view, label)
        if (i + 1) % 1000 == 0:
            for member in members:
                member.decay_learning_rate(i / len(train_x))

    per_member = [0] * len(members)
    ensemble_correct = 0
    disagreements = 0
    rescued = 0
    for i in range(len(test_x)):
        views = encode(test_x[i])
        label = int(test_y[i])
        readings = [m.predict(v) for m, v in zip(members, views, strict=True)]
        for k, reading in enumerate(readings):
            if reading.label == label:
                per_member[k] += 1
        pooled = np.sum([r.distribution for r in readings], axis=0)
        verdict = int(np.argmax(pooled))
        if len({r.label for r in readings}) > 1:
            disagreements += 1
        if verdict == label:
            ensemble_correct += 1
            if readings[0].label != label:
                rescued += 1

    n = len(test_x)
    return Result(
        name=name,
        best_single=max(per_member) / n * 100,
        ensemble=ensemble_correct / n * 100,
        disagreements=disagreements,
        rescued=rescued,
        samples=n,
        seconds=time.time() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="the whole corpus")
    args = parser.parse_args()

    train_n, test_n = (60_000, 10_000) if args.full else (6_000, 2_000)
    data = load(train_n, test_n)
    optics = FourierOptics(shape=(28, 28))
    bands = RadialBands(shape=(28, 28))

    def same_features(image):
        features = optics.apply(image.reshape(28, 28))
        return [features, features, features]

    def split_bands(image):
        stacked = bands.apply(image.reshape(28, 28))
        return list(np.split(stacked, 3))

    def trio(phase_init, n_inputs):
        return [
            QuantumCortex(
                n_inputs, 10, NEURONS_PER_CLASS,
                config=GOLDEN_CONFIG, seed=seed, phase_init=phase_init,
            )
            for seed in (11, 22, 33)
        ]

    print(f"=== ensemble diversity, {train_n} train / {test_n} test ===\n")
    results = [
        measure("identical init (as shipped)", trio("zero", optics.n_features),
                same_features, data),
        measure("random phase init", trio("random", optics.n_features),
                same_features, data),
        measure("three radial bands", trio("random", 784), split_bands, data),
    ]
    for result in results:
        print(result.report())

    print("\nA member count above one is only worth its compute when the")
    print("disagreement column is non-trivial AND the ensemble beats its best")
    print("single member. Check both before switching an ensemble on.")

    # Bagging is the route that survives the phase rule, because the members'
    # training histories genuinely differ rather than their starting points.
    bagged = Ensemble(trio("random", optics.n_features), bag_fraction=0.5, seed=1)
    train_x, train_y, test_x, test_y = data
    for i in range(len(train_x)):
        bagged.observe(optics.apply(train_x[i].reshape(28, 28)), int(train_y[i]))
        if (i + 1) % 1000 == 0:
            bagged.decay_learning_rate(i / len(train_x))
    correct = sum(
        bagged.predict(optics.apply(test_x[i].reshape(28, 28))).label == test_y[i]
        for i in range(len(test_x))
    )
    seen = [m.samples_seen for m in bagged.members]
    print(f"\nbagged (0.5) ensemble: {correct / len(test_x) * 100:.2f}%  members saw {seen}")


if __name__ == "__main__":
    main()
