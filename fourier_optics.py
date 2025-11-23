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