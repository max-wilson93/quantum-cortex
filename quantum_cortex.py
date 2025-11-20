import numpy as np
import cmath

class QuantumCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- PHYSICS CONFIG (85% Baseline) ---
        self.learning_rate = 0.025
        self.phase_flexibility = 0.125
        self.input_threshold = 0.35
        self.kerr_constant = 0.5
        
        # --- CORTEX CONFIG ---
        self.time_steps = 3       # How many times the signal bounces
        self.lateral_inhibit = 0.1 # Strength of cross-class competition
        self.lateral_excite = 0.2  # Strength of same-class resonance
        
        # --- ASTROCYTE CONFIG ---
        self.astrocyte_energy = 0.0
        self.astrocyte_gain = 1.0  # Global volume control
        self.target_energy = 5.0   # Desired total network activity
        
        # --- WEIGHTS ---
        # 1. Afferent (Input -> Cortex)
        # Coherent Silence Initialization
        self.W_in = np.ones((num_inputs, self.num_outputs)) * 0.01 * \
                    np.exp(1j * np.zeros((num_inputs, self.num_outputs)))
                    
        # 2. Lateral (Cortex <-> Cortex)
        # Initialize as Identity Matrix (Self-Reflection) plus noise
        # This helps hold memory (Short Term Potentiation)
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def update_astrocyte(self, current_energy):
        """
        GLIAL REGULATION:
        If the Cortex is too loud (Seizure), the Astrocyte absorbs neurotransmitters (reduces Gain).
        If too quiet (Coma), it releases them (increases Gain).
        """
        error = current_energy - self.target_energy
        
        # Damping is fast, Recovery is slow (Safety mechanism)
        if error > 0:
            self.astrocyte_gain -= 0.05 * error # Fast cool-down
        else:
            self.astrocyte_gain -= 0.01 * error # Slow warm-up
            
        # Hard limits to prevent death or explosion
        self.astrocyte_gain = np.clip(self.astrocyte_gain, 0.1, 2.0)

    def process_image(self, feature_vector, label, train=True):
        if feature_vector.ndim > 1: feature_vector = feature_vector.flatten()
        input_wave = self.get_phasic_input(feature_vector)
        
        # Initial State (Silence)
        cortex_state = np.zeros(self.num_outputs, dtype=complex)
        
        # --- TEMPORAL INTEGRATION LOOP (Thinking) ---
        for t in range(self.time_steps):
            # 1. Input Injection (Constant stimulus)
            feedforward = np.dot(input_wave, self.W_in)
            
            # 2. Lateral Recurrence (Feedback)
            feedback = np.dot(cortex_state, self.W_lat)
            
            # 3. Integration
            # State = Input + Feedback
            cortex_state = feedforward + feedback
            
            # 4. Kerr Non-Linearity (Self-Focusing)
            mags = np.abs(cortex_state)
            phases = np.angle(cortex_state)
            kerr_shift = self.kerr_constant * (mags ** 2)
            cortex_state = mags * np.exp(1j * (phases + kerr_shift))
            
            # 5. Astrocyte Regulation (Global Normalization)
            total_energy = np.sum(mags ** 2)
            if train: 
                self.update_astrocyte(total_energy)
            
            # Apply Gain
            cortex_state *= self.astrocyte_gain

        # --- COLLAPSE & READOUT ---
        energies = np.abs(cortex_state) ** 2
        
        class_energies = np.zeros(10)
        for c in range(10):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_energies[c] = np.sum(energies[start:end])
            
        prediction = np.argmax(class_energies)
        
        # --- PLASTICITY (Learning) ---
        if train:
            start_target = label * self.neurons_per_class
            end_target = start_target + self.neurons_per_class
            
            active_inputs = np.where(np.abs(input_wave) > 0.1)[0]
            
            if len(active_inputs) > 0:
                # A. Feedforward Learning (Standard Quantum STDP)
                for n in range(start_target, end_target):
                    w_sub = self.W_in[active_inputs, n]
                    # Align Phase
                    w_phase = np.angle(w_sub)
                    rot = np.exp(-1j * self.phase_flexibility * w_phase)
                    w_sub *= rot
                    # Grow
                    w_sub *= (1.0 + self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

                # B. Lateral Learning (Hebbian Clustering)
                # "Neurons that resonate together, wire together"
                # We want neurons in the SAME class to excite each other.
                # We want neurons in DIFFERENT classes to inhibit.
                
                # This is computationally expensive (N^2), so we approximate.
                # We boost the diagonal block of the target class in W_lat.
                
                # Get the sub-matrix for the target class
                target_block = self.W_lat[start_target:end_target, start_target:end_target]
                
                # Increase magnitude (Stronger coupling)
                target_block *= (1.0 + self.learning_rate)
                self.W_lat[start_target:end_target, start_target:end_target] = target_block
            
            # C. Damping (Punish Prediction)
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

            # Normalization
            # Feedforward Weights
            mags = np.abs(self.W_in)
            phases = np.angle(self.W_in)
            mags = np.clip(mags, 0.0, 1.0)
            self.W_in = mags * np.exp(1j * phases)
            
            # Lateral Weights
            mags_lat = np.abs(self.W_lat)
            phases_lat = np.angle(self.W_lat)
            mags_lat = np.clip(mags_lat, 0.0, 0.5) # Keep lateral weaker than input
            self.W_lat = mags_lat * np.exp(1j * phases_lat)

        return prediction == label, prediction, self.astrocyte_gain

    def visualize_cortex_ascii(self, digit_idx):
        neuron_idx = digit_idx * self.neurons_per_class
        w_vec = self.W_in[:, neuron_idx]
        w_vec = w_vec[:784]
        mags = np.abs(w_vec)
        phases = np.angle(w_vec)
        
        print(f"\n--- Cortical State (Ch 0) Output {digit_idx} ---")
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