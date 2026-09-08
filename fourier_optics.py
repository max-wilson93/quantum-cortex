"""Fourier-optical front-end: a bank of oriented bandpass filters.

Each mask selects a single-sided angular wedge of the spectrum, bandpassed to
radius 1..14. Because a real image has a Hermitian spectrum, a single-sided
wedge makes each filtered output an *analytic signal*: ``np.abs()`` below is
therefore its phase-invariant envelope, and the phase being discarded is the
local Gabor phase encoding edge position and polarity. Roadmap 1.3 is about
that discarded phase.

Phase 0 leaves ``apply`` untouched -- the front-end is the strong part of this
project. ``apply_batch`` is added purely for speed and is checked against
``apply`` for exact equality in ``tests/test_optics.py``.
"""

import numpy as np
import numpy.fft as fft


class FourierOptics:
    def __init__(self, shape=(28, 28)):
        self.rows, self.cols = shape
        self.masks = []
        self.create_spectral_filters()

    def create_spectral_filters(self):
        crow, ccol = self.rows // 2, self.cols // 2
        y, x = np.ogrid[-crow:self.rows-crow, -ccol:self.cols-ccol]
        theta = np.arctan2(y, x)
        
        target_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        bandwidth = np.pi / 8 
        
        for target in target_angles:
            diff = np.abs(theta - target)
            diff = np.minimum(diff, 2*np.pi - diff)
            radius = np.sqrt(x**2 + y**2)
            # Bandpass filter: Block DC (0-1) and High Freq (>14)
            mask = (diff < bandwidth) & (radius > 1) & (radius < 14)
            self.masks.append(mask)

    def apply(self, image):
        f_transform = fft.fftshift(fft.fft2(image))
        features = []
        for mask in self.masks:
            filtered_f = f_transform * mask
            spatial_result = fft.ifft2(fft.ifftshift(filtered_f))
            magnitude = np.abs(spatial_result)
            if np.max(magnitude) > 0:
                magnitude /= np.max(magnitude)
            features.append(magnitude.flatten())
        return np.concatenate(features)

    def apply_batch(self, images, chunk_size=2048):
        """Vectorised ``apply`` over a stack of images.

        Returns an ``(n_images, n_masks * rows * cols)`` array whose rows equal
        ``apply(image)`` exactly; ``tests/test_optics.py`` asserts bit equality.
        Chunked so peak memory stays bounded regardless of ``n_images``.
        """
        images = np.asarray(images, dtype=float).reshape(-1, self.rows, self.cols)
        n_masks = len(self.masks)
        mask_stack = np.stack(self.masks)[None, :, :, :]
        out = np.empty((images.shape[0], n_masks * self.rows * self.cols), dtype=float)

        for start in range(0, images.shape[0], chunk_size):
            block = images[start:start + chunk_size]
            spectrum = fft.fftshift(fft.fft2(block, axes=(1, 2)), axes=(1, 2))
            filtered = spectrum[:, None, :, :] * mask_stack
            spatial = fft.ifft2(fft.ifftshift(filtered, axes=(2, 3)), axes=(2, 3))
            magnitude = np.abs(spatial)
            peak = magnitude.max(axis=(2, 3), keepdims=True)
            np.divide(magnitude, peak, out=magnitude, where=peak > 0)
            out[start:start + block.shape[0]] = magnitude.reshape(block.shape[0], -1)
        return out
