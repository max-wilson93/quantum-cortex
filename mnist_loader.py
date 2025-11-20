import os
import struct
import numpy as np

class LocalMNISTLoader:
    """
    Handles the low-level binary parsing of MNIST files.
    This file is completely independent of the Neural Network logic.
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def load_images(self, filename):
        filepath = os.path.join(self.base_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading images from {filepath}...")
        with open(filepath, 'rb') as f:
            # Read magic number, count, rows, cols
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            # Read all pixel data
            buffer = f.read(num_images * rows * cols)
            data = np.frombuffer(buffer, dtype=np.uint8)
            
            # Reshape and Normalize (0 to 1)
            return data.reshape(num_images, rows * cols) / 255.0

    def load_labels(self, filename):
        filepath = os.path.join(self.base_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading labels from {filepath}...")
        with open(filepath, 'rb') as f:
            # Read magic number, count
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            # Read all label data
            buffer = f.read(num_labels)
            return np.frombuffer(buffer, dtype=np.uint8)