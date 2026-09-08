"""The corruptions in tasks.py decide whether a robustness claim means anything.

A translation that silently wraps around, or a "noise" level that leaves the
image untouched, would produce a robustness curve that looks fine and says
nothing. These are cheap tests for exactly that.
"""

import numpy as np
import pytest

import tasks


def centre_of_mass(flat):
    images = flat.reshape(-1, 28, 28)
    grid = np.indices((28, 28))
    return np.array([np.average(grid, axis=(1, 2), weights=image) for image in images])


def test_zero_level_corruptions_are_identity(split):
    images = split.images_test[:16]
    rng = np.random.default_rng(0)
    assert np.array_equal(tasks.translate(images, 0), images)
    assert np.array_equal(tasks.add_noise(images, 0.0, rng), images)
    assert np.array_equal(tasks.blur(images, 0.0), images)


def test_translation_moves_the_digit_by_the_requested_amount(split):
    images = split.images_test[:16]
    for pixels in (2, 4):
        shifted = tasks.translate(images, pixels)
        delta = np.mean(centre_of_mass(shifted) - centre_of_mass(images), axis=0)
        assert np.allclose(delta, pixels, atol=0.5), f"shift {pixels} moved by {delta}"


def test_translation_does_not_wrap_around(split):
    """A wrap would keep every pixel on the canvas and make the task easier than
    it looks. Zero fill means a large shift genuinely loses ink off the edge."""
    images = split.images_test[:16]
    assert tasks.translate(images, 10).sum() < images.sum() * 0.9


def test_noise_increases_with_sigma_and_stays_in_range(split):
    images = split.images_test[:16]
    rng = np.random.default_rng(0)
    previous = 0.0
    for sigma in (0.1, 0.3, 0.5):
        noisy = tasks.add_noise(images, sigma, rng)
        assert noisy.min() >= 0.0 and noisy.max() <= 1.0
        deviation = float(np.mean(np.abs(noisy - images)))
        assert deviation > previous
        previous = deviation


def test_blur_reduces_peak_contrast(split):
    images = split.images_test[:16]
    sharp = float(images.std())
    assert float(tasks.blur(images, 1.0).std()) < sharp
    assert float(tasks.blur(images, 2.0).std()) < float(tasks.blur(images, 1.0).std())


def test_corruptions_preserve_shape(split):
    images = split.images_test[:8]
    rng = np.random.default_rng(0)
    for corrupted in (tasks.translate(images, 3), tasks.add_noise(images, 0.2, rng),
                      tasks.blur(images, 1.0)):
        assert corrupted.shape == images.shape


def test_degradation_rate_is_positive_for_a_degrading_curve():
    """Rate, not slope: positive means accuracy is being lost."""
    assert tasks.degradation_rate((0, 1, 2, 3), np.array([90.0, 80.0, 70.0, 60.0])) == pytest.approx(10.0)
    assert tasks.degradation_rate((0, 1, 2, 3), np.array([90.0, 90.0, 90.0, 90.0])) == pytest.approx(0.0)


def test_the_more_robust_model_gets_the_favourable_verdict():
    """Regression test for a sign inversion that reported every robustness
    verdict backwards.

    Both slopes are negative, so "larger slope" and "degrades less" point in
    opposite directions, and comparing the slopes directly flipped the answer:
    a model holding 76% where the other had fallen to 33% was reported as
    degrading *faster*. Rates make the comparison monotone, and this pins it.
    """
    levels = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    robust = [np.array([88.0, 88.0, 87.0, 85.0, 81.0, 76.0])] * 3
    fragile = [np.array([91.0, 88.0, 73.0, 56.0, 43.0, 33.0])] * 3

    assert "cortex degrades more slowly" in tasks.curve_table(
        "t", levels, "Sigma", robust, fragile)
    assert "cortex degrades faster" in tasks.curve_table(
        "t", levels, "Sigma", fragile, robust)


def test_slope_is_negative_for_a_degrading_curve():
    assert tasks._slope((0, 1, 2, 3), np.array([90.0, 80.0, 70.0, 60.0])) == pytest.approx(-10.0)
    assert tasks._slope((0, 1, 2, 3), np.array([90.0, 90.0, 90.0, 90.0])) == pytest.approx(0.0)


def test_matched_hidden_units_matches_the_cortex_parameter_count():
    """The MLP in the class-incremental comparison must not win or lose on
    capacity, so its width is derived from the cortex's parameter count."""
    num_features, num_classes, neurons = 3136, 10, 5
    cortex_parameters = 2 * num_features * num_classes * neurons
    hidden = tasks.matched_hidden_units(num_features, num_classes, cortex_parameters)
    mlp_parameters = num_features * hidden + hidden + hidden * num_classes + num_classes
    assert abs(mlp_parameters - cortex_parameters) / cortex_parameters < 0.02


def test_class_blocks_cover_every_class_exactly_once():
    flat = [c for block in tasks.CLASS_BLOCKS for c in block]
    assert sorted(flat) == list(range(10))
