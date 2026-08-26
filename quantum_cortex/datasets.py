"""Reading the MNIST corpus off disk, for the benchmark that guards the physics.

Nothing here knows anything about the network, and nothing in the network
imports it -- MNIST is the regression fixture, not a dependency of the library.
"""

from __future__ import annotations

import os
import struct

import numpy as np

__all__ = ["LocalMNISTLoader"]


class LocalMNISTLoader:
    """Low-level binary parsing of the IDX files.

    Completely independent of the network logic.
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    def load_images(self, filename: str) -> np.ndarray:
        """Flattened images, normalised to [0, 1]. Shape ``(n, rows * cols)``."""
        filepath = os.path.join(self.base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading images from {filepath}...")
        with open(filepath, "rb") as handle:
            _magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
            buffer = handle.read(count * rows * cols)
            data = np.frombuffer(buffer, dtype=np.uint8)
            return np.asarray(data.reshape(count, rows * cols) / 255.0)

    def load_labels(self, filename: str) -> np.ndarray:
        """Labels as ``uint8``. Shape ``(n,)``."""
        filepath = os.path.join(self.base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading labels from {filepath}...")
        with open(filepath, "rb") as handle:
            _magic, count = struct.unpack(">II", handle.read(8))
            return np.frombuffer(handle.read(count), dtype=np.uint8)