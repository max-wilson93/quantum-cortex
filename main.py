"""The MNIST benchmark. This file is the regression guard.

The published result is 90.74% test accuracy against 90.30% training accuracy,
on 60,000 training samples presented **once each**, with no backpropagation.
Test above train is the part worth staring at: the model generalises better
than it fits, which is not a thing gradient-descended networks normally do.

Run it after any change to the physics or the learning rule:

    python main.py                 # the full published run, ~25 minutes
    python main.py --quick         # 6000/2000, ~90 seconds, for a smoke test
    python main.py --ensemble      # three cortices instead of one

The ``--ensemble`` flag exists to be measured, not to be trusted. See
``quantum_cortex/ensemble.py`` for what it was found to be worth.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np

from quantum_cortex import Ensemble, FourierOptics, QuantumCortex
from quantum_cortex.cortex import GOLDEN_CONFIG
from quantum_cortex.datasets import LocalMNISTLoader

DATA_PATH = "./mnist_data"
LOG_FILE = "quantum_validation_log.csv"
NEURONS_PER_CLASS = 5


def log_experiment(
    train_acc: float, test_acc: float, duration: float, config: dict[str, float], notes: str
) -> None:
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["Timestamp", "Train_Acc", "Test_Acc", "Duration", "Config", "Notes"])
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{train_acc:.2f}%",
                f"{test_acc:.2f}%",
                f"{duration:.1f}s",
                str(config),
                notes,
            ]
        )
    print(f"\n[Log] Validation results saved to {LOG_FILE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="6000 train / 2000 test instead of the full corpus",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="three cortices with distinct seeds rather than one",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seeds the model and the shuffle"
    )
    parser.add_argument("--no-log", action="store_true", help="skip writing the CSV log")
    return parser


def run(args: argparse.Namespace) -> tuple[float, float]:
    train_samples = 6_000 if args.quick else 60_000
    test_samples = 2_000 if args.quick else 10_000

    loader = LocalMNISTLoader(DATA_PATH)
    print(f"--- Quantum Cortex Validation Run (seed {args.seed}) ---")
    print(f"Physics: {GOLDEN_CONFIG}")

    train_images = loader.load_images("train-images.idx3-ubyte")
    train_labels = loader.load_labels("train-labels.idx1-ubyte")
    test_images = loader.load_images("t10k-images.idx3-ubyte")
    test_labels = loader.load_labels("t10k-labels.idx1-ubyte")

    # Seeded so the run is reproducible. The original shuffled from numpy's
    # global RNG, which meant no two runs were comparable.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_images))[:train_samples]
    train_images, train_labels = train_images[order], train_labels[order]
    test_images, test_labels = test_images[:test_samples], test_labels[:test_samples]

    optics = FourierOptics(shape=(28, 28))
    model: QuantumCortex | Ensemble
    if args.ensemble:
        print("-> Initializing Quantum Trinity (three cortices)...")
        model = Ensemble.trinity(
            optics.n_features,
            num_classes=10,
            neurons_per_class=NEURONS_PER_CLASS,
            config=GOLDEN_CONFIG,
            seeds=(args.seed, args.seed + 11, args.seed + 22),
        )
    else:
        print("-> Initializing single cortex...")
        model = QuantumCortex(
            optics.n_features,
            num_classes=10,
            neurons_per_class=NEURONS_PER_CLASS,
            config=GOLDEN_CONFIG,
            seed=args.seed,
        )

    print(f"\n=== PHASE 1: TRAINING ({train_samples} samples, one pass) ===")
    start = time.time()
    correct = 0
    for i in range(train_samples):
        features = optics.apply(train_images[i].reshape(28, 28))
        prediction = model.observe(features, int(train_labels[i]))
        if prediction.label == train_labels[i]:
            correct += 1
        if (i + 1) % 1000 == 0:
            model.decay_learning_rate(i / train_samples)
            print(f"Train {i + 1} | Acc: {correct / (i + 1) * 100:.2f}%")
    train_acc = correct / train_samples * 100
    print(f"Training complete. Final train accuracy: {train_acc:.2f}%")

    print(f"\n=== PHASE 2: VALIDATION ({test_samples} samples) ===")
    print("Plasticity OFF. Testing generalization...")
    test_correct = 0
    low_margin = 0
    for i in range(test_samples):
        features = optics.apply(test_images[i].reshape(28, 28))
        prediction = model.predict(features)
        if prediction.label == test_labels[i]:
            test_correct += 1
        if not prediction.confident(min_margin=0.05):
            low_margin += 1
        if (i + 1) % 1000 == 0:
            print(f"Test {i + 1} | Current test acc: {test_correct / (i + 1) * 100:.2f}%")
    test_acc = test_correct / test_samples * 100
    duration = time.time() - start

    print("\n=== FINAL RESULTS ===")
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Test accuracy:     {test_acc:.2f}%")
    print(f"Generalization gap: {test_acc - train_acc:+.2f} points")
    print(f"Low-margin (<0.05) predictions: {low_margin}/{test_samples}")
    print(f"Duration: {duration:.1f}s")

    if not args.no_log:
        notes = "quick" if args.quick else "full"
        if args.ensemble:
            notes += " ensemble"
        log_experiment(train_acc, test_acc, duration, GOLDEN_CONFIG, f"Validation run ({notes})")

    return train_acc, test_acc


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
