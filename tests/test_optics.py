"""The front-end is the strong part of this project. Do not let it drift.

`apply_batch` exists only for speed. If it ever stops matching `apply` exactly,
every feature-based number in results.md silently changes meaning.
"""

import numpy as np

from fourier_optics import FourierOptics


def test_batch_matches_single_image_exactly(split):
    optics = FourierOptics(shape=(28, 28))
    images = split.images_train[:64]
    one_at_a_time = np.array([optics.apply(x.reshape(28, 28)) for x in images])
    batched = optics.apply_batch(images)
    assert np.array_equal(one_at_a_time, batched)


def test_batch_chunking_is_irrelevant(split):
    optics = FourierOptics(shape=(28, 28))
    images = split.images_train[:50]
    assert np.array_equal(optics.apply_batch(images, chunk_size=7),
                          optics.apply_batch(images, chunk_size=1000))


def test_features_are_normalised_per_channel(split):
    optics = FourierOptics(shape=(28, 28))
    features = optics.apply(split.images_train[0].reshape(28, 28))
    assert features.shape == (4 * 28 * 28,)
    for channel in features.reshape(4, -1):
        assert 0.0 <= channel.min() and channel.max() <= 1.0


def test_masks_are_single_sided_wedges():
    """Each mask covers a one-sided angular wedge, which is what makes every
    filtered output an analytic signal -- and therefore what makes the
    discarded phase a local Gabor phase (roadmap 1.3)."""
    optics = FourierOptics(shape=(28, 28))
    for mask in optics.masks:
        assert mask.any(), "mask selects nothing"
        # A two-sided (Hermitian-symmetric) mask would equal its own 180-degree
        # rotation. These must not.
        assert not np.array_equal(mask, mask[::-1, ::-1])
