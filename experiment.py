"""One function, config in, accuracy out.

Every script in this repository -- ``main.py``, ``bench.py``, ``ablate.py`` and
the tests -- routes through :func:`run_experiment`. There is exactly one
training loop and one evaluation loop in this project, so an ablation differs
from the full model only in the config it is handed.
"""

import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from quantum_cortex import MECHANISMS, QuantumCortex


@dataclass(frozen=True)
class ModelConfig:
    """A complete description of one model. Hashable, printable, diffable."""

    # structure
    ensemble_size: int = 3
    neurons_per_class: int = 5

    # physics constants
    learning_rate: float = 0.09
    phase_flexibility: float = 0.1
    lateral_strength: float = 0.16
    input_threshold: float = 0.7
    kerr_constant: float = 0.2
    system_energy: float = 40.0
    time_steps: int = 4

    # mechanism switches (Phase 0.4)
    lateral_coupling: bool = True
    recurrence: bool = True
    kerr: bool = True
    phase_input: bool = True
    energy_clamp: bool = True

    # initialisation and plasticity
    init: str = "uniform"
    train: bool = True          # False = the untrained-cortex control
    anneal: bool = True

    def cortex_config(self):
        """The subset of this config that ``QuantumCortex`` accepts."""
        keys = ("learning_rate", "phase_flexibility", "lateral_strength",
                "input_threshold", "kerr_constant", "system_energy",
                "time_steps", "init") + MECHANISMS
        return {k: getattr(self, k) for k in keys}

    def differences_from(self, other):
        """Fields where this config differs from ``other``, for table labels."""
        mine, theirs = asdict(self), asdict(other)
        return {k: v for k, v in mine.items() if theirs[k] != v}

    def with_(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class RunResult:
    """What one seeded run produced. Accuracies are percentages."""

    config: ModelConfig
    seed: int
    n_train: int
    n_test: int

    #: Running accuracy of the ensemble vote *while it was learning*. Includes
    #: the untrained warm-up, so it is systematically pessimistic. This is the
    #: number the original README compared against test accuracy to claim "zero
    #: overfitting"; it is reported here but is never a decision metric.
    online_accuracy: float

    #: Accuracy on the same training samples with plasticity off, measured
    #: after learning finished. This is the honest train accuracy (Phase 0.1).
    train_accuracy: float

    test_accuracy: float
    train_seconds: float
    eval_seconds: float
    test_predictions: np.ndarray


def build_ensemble(config, num_features, num_classes, seed):
    """Seed each ensemble member independently from one run seed (Phase 0.2)."""
    children = np.random.SeedSequence(seed).spawn(config.ensemble_size)
    return [
        QuantumCortex(num_features, num_classes, config.neurons_per_class,
                      config=config.cortex_config(), seed=child)
        for child in children
    ]


def vote(predictions, num_classes):
    """Majority vote, ties broken toward the lowest class index (as shipped)."""
    counts = np.zeros(num_classes)
    for p in predictions:
        counts[p] += 1
    return int(np.argmax(counts))


def _pass(ensemble, features, labels, num_classes, train, config=None,
          verbose=False, tag="", log_every=1000):
    """One pass over a dataset. Returns (accuracy_pct, predictions)."""
    n = len(labels)
    predictions = np.empty(n, dtype=np.int64)
    correct = 0

    for i in range(n):
        member_preds = [c.process_image(features[i], labels[i], train=train)[1]
                        for c in ensemble]
        predictions[i] = vote(member_preds, num_classes)
        if predictions[i] == labels[i]:
            correct += 1

        if train and config is not None and config.anneal and (i + 1) % 1000 == 0:
            progress = i / n
            for c in ensemble:
                c.decay_learning_rate(progress)

        if verbose and (i + 1) % log_every == 0:
            print(f"  {tag} {i + 1}/{n} | running acc {(correct / (i + 1)) * 100:.2f}%",
                  flush=True)

    return (correct / n) * 100.0, predictions


def run_experiment(config, split, seed=None, verbose=False, log_every=1000):
    """Train (optionally), then measure train and test accuracy with plasticity off.

    The three accuracies returned are deliberately kept apart:

    * ``online_accuracy`` -- accuracy while learning, warm-up included;
    * ``train_accuracy`` -- a second pass over the *same* training samples with
      plasticity off;
    * ``test_accuracy`` -- held-out, plasticity off. The only decision metric.

    Comparing the first against the third is what produced the original "test >
    train, zero overfitting" claim. Comparing the second against the third is
    the comparison that actually means something.
    """
    seed = split.seed if seed is None else seed
    ensemble = build_ensemble(config, split.num_features, split.num_classes, seed)

    start = time.time()
    if config.train:
        online_accuracy, _ = _pass(ensemble, split.features_train, split.labels_train,
                                   split.num_classes, train=True, config=config,
                                   verbose=verbose, tag="train", log_every=log_every)
    else:
        # The untrained control never sees a gradient of any kind; there is no
        # "during learning" number to report.
        online_accuracy = float("nan")
    train_seconds = time.time() - start

    start = time.time()
    train_accuracy, _ = _pass(ensemble, split.features_train, split.labels_train,
                              split.num_classes, train=False)
    test_accuracy, test_predictions = _pass(ensemble, split.features_test,
                                            split.labels_test, split.num_classes,
                                            train=False)
    eval_seconds = time.time() - start

    return RunResult(
        config=config,
        seed=seed,
        n_train=split.n_train,
        n_test=split.n_test,
        online_accuracy=online_accuracy,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        test_predictions=test_predictions,
    )


def weight_saturation(cortex):
    """How much of the magnitude field has been driven to its bounds.

    ``w *= (1 + lr)`` followed by ``clip(0, 1)`` pushes weights toward the ends
    of the range. This quantifies it; roadmap 3.2 replaces the rule.
    """
    mags = np.abs(cortex.W_in)
    return {
        "pinned_at_max": float(np.mean(mags >= 1.0 - 1e-12) * 100),
        "never_moved": float(np.mean(np.isclose(mags, 0.05)) * 100),
        "mean": float(mags.mean()),
    }
