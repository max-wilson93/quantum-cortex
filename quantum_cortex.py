import numpy as np
import cmath

class QuantumCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class, config=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- HYPERPARAMETERS (Injectable for Monte Carlo) ---
        if config:
            self.learning_rate = config.get('learning_rate', 0.025)
            self.phase_flexibility = config.get('phase_flexibility', 0.125)
            self.lateral_strength = config.get('lateral_strength', 0.2)
            self.input_threshold = config.get('input_threshold', 0.35)
            self.kerr_constant = config.get('kerr_constant', 0.5)
            self.system_energy = config.get('system_energy', 10.0)
        else:
            # The "Golden" Defaults (86% Run)
            self.learning_rate = 0.025
            self.phase_flexibility = 0.125
            self.lateral_strength = 0.2
            self.input_threshold = 0.35
            self.kerr_constant = 0.5
            self.system_energy = 10.0
        
        self.time_steps = 4
        
        # --- WEIGHTS ---
        # Standard fixed allocation
        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.05
        self.W_in = init_mag * np.exp(1j * np.zeros((num_inputs, self.num_outputs)))
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def normalize_state(self, state_vector):
        """
        UNITARY L2 NORMALIZATION
        The core stabilizer of the 86% model.
        """
        current_energy = np.linalg.norm(state_vector)
        if current_energy > 0:
            scale = self.system_energy / current_energy
            if scale < 1.0: # Only damp, never amplify noise
                state_vector *= scale
        return state_vector

    def process_image(self, feature_vector, label, train=True):
        if feature_vector.ndim > 1: feature_vector = feature_vector.flatten()
        input_wave = self.get_phasic_input(feature_vector)
        
        cortex_state = np.zeros(self.num_outputs, dtype=complex)
        
        # --- RESONANT LOOP ---
        for t in range(self.time_steps):
            # 1. Drive
            feedforward = np.dot(input_wave, self.W_in)
            # 2. Coupling
            feedback = np.dot(cortex_state, self.W_lat)
            # 3. Integrate
            cortex_state = feedforward + feedback
            
            # 4. Kerr Non-Linearity
            mags = np.abs(cortex_state)
            phases = np.angle(cortex_state)
            kerr_shift = self.kerr_constant * (mags ** 2)
            cortex_state = mags * np.exp(1j * (phases + kerr_shift))
            
            # 5. L2 Normalization
            cortex_state = self.normalize_state(cortex_state)

        # --- READOUT ---
        energies = np.abs(cortex_state) ** 2
        total_energy = np.sum(energies)
        
        class_energies = np.zeros(10)
        for c in range(10):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_energies[c] = np.sum(energies[start:end])
            
        prediction = np.argmax(class_energies)
        
        # --- LEARNING ---
        if train:
            start_target = label * self.neurons_per_class
            end_target = start_target + self.neurons_per_class
            active_inputs = np.where(np.abs(input_wave) > 0.1)[0]
            
            if len(active_inputs) > 0:
                # A. Resonance (Feedforward)
                for n in range(start_target, end_target):
                    w_sub = self.W_in[active_inputs, n]
                    w_phase = np.angle(w_sub)
                    # Rotate to 0
                    rot = np.exp(-1j * self.phase_flexibility * w_phase)
                    w_sub *= rot
                    # Grow
                    w_sub *= (1.0 + self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

                # B. Lateral Clustering (Simple Hebbian)
                # Boost target block (Self-Excitatory for Class)
                target_block = self.W_lat[start_target:end_target, start_target:end_target]
                target_block *= (1.0 + self.learning_rate)
                self.W_lat[start_target:end_target, start_target:end_target] = target_block
            
            # C. Damping
            if prediction != label:
                start_wrong = prediction * self.neurons_per_class
                end_wrong = start_wrong + self.neurons_per_class
                for n in range(start_wrong, end_wrong):
                    w_sub = self.W_in[active_inputs, n]
                    noise = np.random.uniform(-1.0, 1.0, size=len(active_inputs))
                    dec = np.exp(1j * self.phase_flexibility * noise)
                    w_sub *= dec
                    w_sub *= (1.0 - self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

            # Normalize Weights (L2)
            mags = np.abs(self.W_in)
            phases = np.angle(self.W_in)
            mags = np.clip(mags, 0.0, 1.0)
            self.W_in = mags * np.exp(1j * phases)
            
            mags_lat = np.abs(self.W_lat)
            phases_lat = np.angle(self.W_lat)
            mags_lat = np.clip(mags_lat, 0.0, 0.5)
            self.W_lat = mags_lat * np.exp(1j * phases_lat)

        return prediction == label, prediction, total_energy