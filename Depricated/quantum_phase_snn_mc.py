import numpy as np
import random
import cmath

class QuantumPhaseSNN:
    def __init__(self, num_inputs, num_classes, neurons_per_class, config):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- HYPERPARAMETERS (Injected) ---
        self.learning_rate = config.get('learning_rate', 0.1)
        self.phase_flexibility = config.get('phase_flexibility', 0.2)
        self.lateral_strength = config.get('lateral_strength', 0.1)
        self.input_threshold = config.get('input_threshold', 0.4)
        
        # --- INITIALIZATION ---
        # Mag: 0.01 (Silence)
        # Phase: 0.0 (Coherent)
        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.01
        init_phase = np.zeros((num_inputs, self.num_outputs)) 
        
        self.weights = init_mag * np.exp(1j * init_phase)

    def get_phasic_input(self, image_data):
        # Dynamic Thresholding based on config
        mag = np.where(image_data > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def process_image(self, image_data, label, train=True):
        if image_data.ndim > 1: image_data = image_data.flatten()
        input_wave = self.get_phasic_input(image_data)
        
        # 1. INTERFERENCE
        output_wave = np.dot(input_wave, self.weights)
        
        # 2. LATERAL PHASE-LOCKING (Coupling)
        if self.lateral_strength > 0:
            class_waves = output_wave.reshape(self.num_classes, self.neurons_per_class)
            mean_fields = np.mean(class_waves, axis=1)
            coherence_signal = np.repeat(mean_fields, self.neurons_per_class)
            output_wave += coherence_signal * self.lateral_strength
        
        # 3. OBSERVATION
        energies = np.abs(output_wave) ** 2
        
        # 4. COLLAPSE
        class_energies = np.zeros(10)
        for c in range(10):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_energies[c] = np.sum(energies[start:end])
            
        prediction = np.argmax(class_energies)
        
        if train:
            # A. RESONANCE (Target)
            start_target = label * self.neurons_per_class
            end_target = start_target + self.neurons_per_class
            
            active_indices = np.where(np.abs(input_wave) > 0.1)[0]
            
            if len(active_indices) > 0:
                for n in range(start_target, end_target):
                    w_sub = self.weights[active_indices, n]
                    
                    # Rotate
                    current_phase = np.angle(w_sub)
                    rotation = np.exp(-1j * self.phase_flexibility * current_phase)
                    w_sub *= rotation
                    
                    # Grow
                    w_sub *= (1.0 + self.learning_rate)
                    self.weights[active_indices, n] = w_sub

            # B. DAMPING (Wrong Prediction)
            if prediction != label:
                start_wrong = prediction * self.neurons_per_class
                end_wrong = start_wrong + self.neurons_per_class
                
                for n in range(start_wrong, end_wrong):
                    w_sub = self.weights[active_indices, n]
                    
                    # Scramble
                    noise = np.random.uniform(-1.0, 1.0, size=len(active_indices))
                    decoherence = np.exp(1j * self.phase_flexibility * noise)
                    w_sub *= decoherence
                    
                    # Decay
                    w_sub *= (1.0 - self.learning_rate)
                    self.weights[active_indices, n] = w_sub

            # NORMALIZE
            mags = np.abs(self.weights)
            phases = np.angle(self.weights)
            mags = np.clip(mags, 0.0, 1.0)
            self.weights = mags * np.exp(1j * phases)

        return prediction == label, prediction