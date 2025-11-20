import numpy as np
import numpy.fft as fft

class FourierOptics:
    def __init__(self, shape=(28, 28)):
        self.rows, self.cols = shape
        self.masks = []
        
        # Create Frequency Domain Masks (The "Filter" in the 4f system)
        # We want 4 orientations: 0, 45, 90, 135 degrees.
        # In Fourier space, these look like "Wedges" or "Pie Slices" 
        # passing through the center (DC component).
        
        self.create_spectral_filters()

    def create_spectral_filters(self):
        # Create a coordinate grid for frequency space
        crow, ccol = self.rows // 2, self.cols // 2
        y, x = np.ogrid[-crow:self.rows-crow, -ccol:self.cols-ccol]
        
        # Calculate Angle of every frequency pixel
        # theta is -pi to pi
        theta = np.arctan2(y, x)
        
        # Create 4 directional masks (Wedges)
        # Bandwidth of +/- 22.5 degrees around the target angle
        target_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        bandwidth = np.pi / 8 
        
        for target in target_angles:
            # Angular distance logic
            diff = np.abs(theta - target)
            # Handle wrapping (pi and -pi are same)
            diff = np.minimum(diff, 2*np.pi - diff)
            
            # Create Binary Mask: 1 if inside wedge, 0 if outside
            # We also block the very center (DC) to remove "flat" background (High Pass)
            radius = np.sqrt(x**2 + y**2)
            mask = (diff < bandwidth) & (radius > 1)
            
            self.masks.append(mask)

    def apply(self, image):
        """
        Performs the 4f Correlation:
        Space -> FFT -> Mask -> IFFT -> Space
        """
        # 1. FFT (Move to Frequency/Momentum Space)
        # fftshift moves the DC component (0 frequency) to the center
        f_transform = fft.fftshift(fft.fft2(image))
        
        features = []
        
        for mask in self.masks:
            # 2. Spectral Filtering (The "Holographic Mask")
            # Multiply the wave by the mask
            filtered_f = f_transform * mask
            
            # 3. IFFT (Move back to Real Space)
            # ifftshift moves center back to corners for correct transform
            spatial_result = fft.ifft2(fft.ifftshift(filtered_f))
            
            # 4. Discretization (Born Rule)
            # The result is Complex. We want the Magnitude (Energy).
            magnitude = np.abs(spatial_result)
            
            # Normalize
            if np.max(magnitude) > 0:
                magnitude /= np.max(magnitude)
                
            features.append(magnitude.flatten())
            
        return np.concatenate(features)