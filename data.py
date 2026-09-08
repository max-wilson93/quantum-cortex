"""Seeded, reproducible data preparation shared by main, bench, ablate and tests.

Every experiment in this repository goes through :func:`make_split`, so the
sample count, the shuffle and the feature extraction are identical across the
model and its baselines. That is the only way an ablation number means
anything.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from fourier_optics import FourierOptics
from mnist_loader import LocalMNISTLoader

DEFAULT_DATA_PATH = "./mnist_data"

_RAW_CACHE = {}
_SPLIT_CACHE = {}


@dataclass
class Split:
    """One seeded train/test split, with pixels and Fourier features."""

    seed: int
    images_train: np.ndarray          # (n_train, 784) pixels in [0, 1]
    labels_train: np.ndarray
    images_test: np.ndarray
    labels_test: np.ndarray
    features_train: np.ndarray        # (n_train, 3136) Fourier-optics features
    features_test: np.ndarray
    num_classes: int = 10
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self):
        return len(self.labels_train)

    @property
    def n_test(self):
        return len(self.labels_test)

    @property
    def num_features(self):
        return self.features_train.shape[1]


def load_raw(data_path=DEFAULT_DATA_PATH):
    """Load the four idx files once per process."""
    key = os.path.abspath(data_path)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    loader = LocalMNISTLoader(data_path)
    try:
        raw = (
            loader.load_images("train-images.idx3-ubyte"),
            loader.load_labels("train-labels.idx1-ubyte"),
            loader.load_images("t10k-images.idx3-ubyte"),
            loader.load_labels("t10k-labels.idx1-ubyte"),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n\nMNIST is no longer committed to this repository. "
            f"Fetch it with:\n    python download_mnist.py"
        ) from None

    _RAW_CACHE[key] = raw
    return raw


def make_split(seed, n_train=None, n_test=None, data_path=DEFAULT_DATA_PATH, cache=True):
    """Build a seeded split and its Fourier features.

    The seed drives the shuffle of both the train and the test set, so a subset
    of size ``n_train`` is a different (but equally valid) sample of MNIST for
    each seed. That is where the seed-to-seed variance in the reported tables
    comes from, alongside per-cortex weight and noise seeding.

    The official train/test boundary is never crossed: test images only ever
    come from ``t10k-*``.
    """
    key = (os.path.abspath(data_path), int(seed), n_train, n_test)
    if cache and key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]

    train_images, train_labels, test_images, test_labels = load_raw(data_path)

    rng = np.random.default_rng(seed)
    train_order = rng.permutation(len(train_images))[: n_train or len(train_images)]
    test_order = rng.permutation(len(test_images))[: n_test or len(test_images)]

    images_train = train_images[train_order]
    labels_train = train_labels[train_order]
    images_test = test_images[test_order]
    labels_test = test_labels[test_order]

    optics = FourierOptics(shape=(28, 28))
    split = Split(
        seed=int(seed),
        images_train=images_train,
        labels_train=labels_train,
        images_test=images_test,
        labels_test=labels_test,
        features_train=optics.apply_batch(images_train),
        features_test=optics.apply_batch(images_test),
        meta={"data_path": os.path.abspath(data_path)},
    )
    if cache:
        _SPLIT_CACHE[key] = split
    return split


def clear_cache():
    """Drop cached splits. Only needed by long-running scripts under memory pressure."""
    _SPLIT_CACHE.clear()
