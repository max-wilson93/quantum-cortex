import numpy as np
from quantum_cortex import QuantumCortex

class LogicalCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class):
        print("   > Initializing Alpha Lobe (Low Freq)...")
        self.cortex_low = QuantumCortex(num_inputs, num_classes, neurons_per_class)
        
        print("   > Initializing Beta Lobe (Mid Freq)...")
        self.cortex_mid = QuantumCortex(num_inputs, num_classes, neurons_per_class)
        
        print("   > Initializing Gamma Lobe (High Freq)...")
        self.cortex_high = QuantumCortex(num_inputs, num_classes, neurons_per_class)

    def decay_learning_rate(self, progress):
        self.cortex_low.decay_learning_rate(progress)
        self.cortex_mid.decay_learning_rate(progress)
        self.cortex_high.decay_learning_rate(progress)

    def process_multiband(self, bands, label, train=True):
        f_low, f_mid, f_high = bands
        
        # Train all 3 lobes independently
        _, pred_l, nrg_l = self.cortex_low.process_image(f_low, label, train)
        _, pred_m, nrg_m = self.cortex_mid.process_image(f_mid, label, train)
        _, pred_h, nrg_h = self.cortex_high.process_image(f_high, label, train)
        
        # VOTING LOGIC
        votes = np.zeros(10)
        votes[pred_l] += 1.0 
        votes[pred_m] += 1.0
        votes[pred_h] += 0.5 # Weight High Freq less (it's noisier)
        
        logical_pred = np.argmax(votes)
        is_correct = (logical_pred == label)
        
        avg_energy = (nrg_l + nrg_m + nrg_h) / 3.0
        
        return is_correct, logical_pred, avg_energy

    def visualize(self, digit_idx):
        print(f"\n=== LOGICAL QUBIT STATE (Class {digit_idx}) ===")
        print("Alpha Lobe (Structure):")
        self.cortex_low.visualize_cortex_ascii(digit_idx)
        print("Beta Lobe (Shape):")
        self.cortex_mid.visualize_cortex_ascii(digit_idx)