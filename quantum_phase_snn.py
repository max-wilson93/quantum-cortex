import numpy as np
import cmath

class QuantumPhaseSNN:
    def __init__(self, num_inputs, num_classes, neurons_per_class, config=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- PHYSICS PARAMETERS ---
        if config:
            self.learning_rate = config.get('learning_rate', 0.025)
            self.phase_flexibility = config.get('phase_flexibility', 0.125)
            self.lateral_strength = config.get('lateral_strength', 0.275)
            self.input_threshold = config.get('input_threshold', 0.35)
            self.kerr_constant = config.get('kerr_constant', 0.5) 
        else:
            self.learning_rate = 0.025
            self.phase_flexibility = 0.125
            self.lateral_strength = 0.275
            self.input_threshold = 0.35
            self.kerr_constant = 0.5
        
        # Initialization
        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.01
        init_phase = np.zeros((num_inputs, self.num_outputs)) 
        self.weights = init_mag * np.exp(1j * init_phase)

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def process_image(self, feature_vector, label, train=True):
        if feature_vector.ndim > 1: feature_vector = feature_vector.flatten()
        input_wave = self.get_phasic_input(feature_vector)
        
        # 1. LINEAR INTERFERENCE
        output_wave = np.dot(input_wave, self.weights)
        
        # 2. QUANTUM NON-LINEARITY (The Optical Kerr Effect)
        # High intensity signals induce a self-phase shift.
        # This separates "Signal" from "Noise" in the phase domain.
        magnitudes = np.abs(output_wave)
        current_phases = np.angle(output_wave)
        
        # Phase Shift = Chi * Intensity (Mag^2)
        kerr_shift = self.kerr_constant * (magnitudes ** 2)
        
        # Apply the twist
        output_wave = magnitudes * np.exp(1j * (current_phases + kerr_shift))
        
        # 3. LATERAL COHERENCE
        if self.lateral_strength > 0:
            class_waves = output_wave.reshape(self.num_classes, self.neurons_per_class)
            mean_fields = np.mean(class_waves, axis=1)
            coherence = np.repeat(mean_fields, self.neurons_per_class)
            output_wave += coherence * self.lateral_strength
        
        # 4. OBSERVATION
        energies = np.abs(output_wave) ** 2
        
        # 5. COLLAPSE
        class_energies = np.zeros(10)
        for c in range(10):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_energies[c] = np.sum(energies[start:end])
            
        prediction = np.argmax(class_energies)
        
        if train:
            start_target = label * self.neurons_per_class
            end_target = start_target + self.neurons_per_class
            active_indices = np.where(np.abs(input_wave) > 0.1)[0]
            
            if len(active_indices) > 0:
                # A. RESONANCE
                for n in range(start_target, end_target):
                    w_sub = self.weights[active_indices, n]
                    # Rotate to 0
                    current_phase = np.angle(w_sub)
                    rotation = np.exp(-1j * self.phase_flexibility * current_phase)
                    w_sub *= rotation
                    # Grow
                    w_sub *= (1.0 + self.learning_rate)
                    self.weights[active_indices, n] = w_sub

            # B. DAMPING
            if prediction != label:
                start_wrong = prediction * self.neurons_per_class
                end_wrong = start_wrong + self.neurons_per_class
                
                for n in range(start_wrong, end_wrong):
                    w_sub = self.weights[active_indices, n]
                    noise = np.random.uniform(-1.0, 1.0, size=len(active_indices))
                    decoherence = np.exp(1j * self.phase_flexibility * noise)
                    w_sub *= decoherence
                    w_sub *= (1.0 - self.learning_rate)
                    self.weights[active_indices, n] = w_sub

            # Normalize
            mags = np.abs(self.weights)
            phases = np.angle(self.weights)
            mags = np.clip(mags, 0.0, 1.0)
            self.weights = mags * np.exp(1j * phases)

        return prediction == label, prediction

    def visualize_phase_ascii(self, digit_idx):
        neuron_idx = digit_idx * self.neurons_per_class
        w_vec = self.weights[:, neuron_idx]
        w_vec = w_vec[:784] 
        mags = np.abs(w_vec)
        phases = np.angle(w_vec)
        print(f"\n--- Kerr Phase Hologram (Ch 0) Output {digit_idx} ---")
        try:
            side = 28
            grid_mag = mags.reshape(side, side)
            grid_phase = phases.reshape(side, side)
            max_mag = np.max(grid_mag)
            if max_mag == 0: max_mag = 1.0
            phase_chars = ['-', '/', '|', '\\'] 
            for r in range(0, side, 2): 
                line = ""
                for c in range(side):
                    m = grid_mag[r, c]
                    p = grid_phase[r, c]
                    if m < (0.2 * max_mag): line += " " 
                    else:
                        p_norm = (p + np.pi) / (2*np.pi)
                        char_idx = int(p_norm * 4) % 4
                        line += phase_chars[char_idx]
                print(line)
        except: pass