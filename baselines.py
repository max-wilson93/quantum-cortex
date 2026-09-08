"""Honest baselines, run on exactly the data the model is run on.

The point of this file is the second entry: **logistic regression on the
Fourier-optics features**. It separates the front-end from the cortex. Without
it, any accuracy the full model reaches is unattributable -- it could be the
spectral filters doing all the work.
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier


@dataclass
class BaselineResult:
    name: str
    train_accuracy: float
    test_accuracy: float
    seconds: float
    converged: bool = True
    note: str = ""


def _fit_and_score(name, model, X_train, y_train, X_test, y_test, note=""):
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return BaselineResult(
        name=name,
        train_accuracy=model.score(X_train, y_train) * 100,
        test_accuracy=model.score(X_test, y_test) * 100,
        seconds=time.time() - start,
        converged=converged,
        note=note if converged else (note + " [did not converge]").strip(),
    )


def logreg_pixels(split, seed, max_iter=1000):
    """Multinomial logistic regression on raw pixels: the no-front-end control."""
    return _fit_and_score(
        "Logistic regression, raw pixels",
        LogisticRegression(max_iter=max_iter, random_state=seed),
        split.images_train, split.labels_train, split.images_test, split.labels_test,
        note=f"{split.images_train.shape[1]}-dim",
    )


def logreg_features(split, seed, max_iter=1000):
    """The critical control: a linear model on the model's own features."""
    return _fit_and_score(
        "Logistic regression, Fourier features",
        LogisticRegression(max_iter=max_iter, random_state=seed),
        split.features_train, split.labels_train, split.features_test, split.labels_test,
        note=f"{split.num_features}-dim",
    )


def nearest_centroid_features(split, seed):
    """One prototype per class, no learning rate, no iterations. The floor for
    a prototype method on these features -- the cortex is a prototype method
    with five prototypes per class, so this is its nearest honest ancestor."""
    return _fit_and_score(
        "Nearest centroid, Fourier features",
        NearestCentroid(),
        split.features_train, split.labels_train, split.features_test, split.labels_test,
    )


def mlp_pixels(split, seed, hidden=(128,), max_iter=200):
    """Upper reference, not a target. A small backprop MLP on raw pixels says
    roughly what the task allows; it is not what this architecture is for."""
    return _fit_and_score(
        f"MLP {hidden}, raw pixels (upper reference)",
        MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=seed),
        split.images_train, split.labels_train, split.images_test, split.labels_test,
        note=f"{max_iter} epochs max",
    )


def linear_readout_predictions(split, seed, max_iter=1000):
    """Fit the linear readout and return its test predictions.

    ``ablate.py`` uses this as the control that matters most: replace the whole
    cortex with a linear readout on the same features and see what is lost.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(split.features_train, split.labels_train)
    predictions = model.predict(split.features_test)
    accuracy = float(np.mean(predictions == split.labels_test) * 100)
    return accuracy, predictions


ALL_BASELINES = (logreg_pixels, logreg_features, nearest_centroid_features, mlp_pixels)
