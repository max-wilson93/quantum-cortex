import numpy as np
import cmath

class QuantumCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- PHYSICS CONFIG (L2 Unitary Mode) ---
        self.learning_rate = 0.025 
        self.phase_flexibility = 0.125
        self.input_threshold = 0.35
        self.kerr_constant = 0.5
        
        self.time_steps = 4
        self.lateral_strength = 0.2
        
        # Target Energy (The "Volume" of the system)
        # We clamp the total energy of the cortex to this value every step.
        self.system_energy = 10.0 
        
        # Weights
        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.05
        self.W_in = init_mag * np.exp(1j * np.zeros((num_inputs, self.num_outputs)))
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def normalize_state(self, state_vector):
        """
        UNITARY OPERATOR (L2 Normalization):
        Preserves contrast but prevents explosion.
        """
        current_energy = np.linalg.norm(state_vector)
        if current_energy > 0:
            scale = self.system_energy / current_energy
            # Only scale DOWN (damping), never up (amplification of noise)
            if scale < 1.0:
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
            
            # 4. Kerr Effect (Phase Twist)
            mags = np.abs(cortex_state)
            phases = np.angle(cortex_state)
            kerr_shift = self.kerr_constant * (mags ** 2)
            cortex_state = mags * np.exp(1j * (phases + kerr_shift))
            
            # 5. UNITARY NORMALIZATION (The L2 Fix)
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
                # A. Resonance (Grow)
                for n in range(start_target, end_target):
                    w_sub = self.W_in[active_inputs, n]
                    w_phase = np.angle(w_sub)
                    rot = np.exp(-1j * self.phase_flexibility * w_phase)
                    w_sub *= rot
                    w_sub *= (1.0 + self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

                # B. Lateral (Cluster)
                target_block = self.W_lat[start_target:end_target, start_target:end_target]
                target_block *= (1.0 + self.learning_rate)
                self.W_lat[start_target:end_target, start_target:end_target] = target_block
            
            # C. Damping (Shrink)
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

            # NORMALIZATION (Weights)
            mags = np.abs(self.W_in)
            phases = np.angle(self.W_in)
            mags = np.clip(mags, 0.0, 1.0)
            self.W_in = mags * np.exp(1j * phases)
            
            mags_lat = np.abs(self.W_lat)
            phases_lat = np.angle(self.W_lat)
            mags_lat = np.clip(mags_lat, 0.0, 0.5)
            self.W_lat = mags_lat * np.exp(1j * phases_lat)

        # --- CORRECT RETURN STATEMENT ---
        # Returns: (Boolean Correct?, Int Prediction, Float Total_Energy)
        return prediction == label, prediction, total_energy