import numpy as np
import cmath

class QuantumCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class, config=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- PHYSICS CONFIG ---
        if config:
            self.learning_rate = config.get('learning_rate', 0.09)
            self.phase_flexibility = config.get('phase_flexibility', 0.1)
            self.lateral_strength = config.get('lateral_strength', 0.16)
            self.input_threshold = config.get('input_threshold', 0.7)
            self.kerr_constant = config.get('kerr_constant', 0.2)
            self.system_energy = config.get('system_energy', 40.0)
        else:
            # The "Golden" Defaults (90+% Run)
            self.learning_rate = 0.09
            self.phase_flexibility = 0.1
            self.lateral_strength = 0.16
            self.input_threshold = 0.7
            self.kerr_constant = 0.2
            self.system_energy = 40.0
        
        # Store initial values for annealing
        self.init_lr = self.learning_rate
        self.init_flex = self.phase_flexibility
        
        self.time_steps = 4
        
        # --- WEIGHTS ---
        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.05
        self.W_in = init_mag * np.exp(1j * np.zeros((num_inputs, self.num_outputs)))
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1

    def decay_learning_rate(self, progress):
        """
        ANNEALING:
        Linearly decays plasticity.
        """
        decay_factor = 1.0 - (progress * 0.9)
        self.learning_rate = self.init_lr * decay_factor
        self.phase_flexibility = self.init_flex * decay_factor

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def normalize_state(self, state_vector):
        """Unitary L2 Normalization (Stability)"""
        current_energy = np.linalg.norm(state_vector)
        if current_energy > 0:
            scale = self.system_energy / current_energy
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
                # 1. FEEDFORWARD (Phase Hebbian)
                for n in range(start_target, end_target):
                    w_sub = self.W_in[active_inputs, n]
                    w_phase = np.angle(w_sub)
                    # Rotate to 0
                    rot = np.exp(-1j * self.phase_flexibility * w_phase)
                    w_sub *= rot
                    # Grow
                    w_sub *= (1.0 + self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

                # 2. LATERAL (Simple Hebbian Clustering)
                target_block = self.W_lat[start_target:end_target, start_target:end_target]
                target_block *= (1.0 + self.learning_rate)
                self.W_lat[start_target:end_target, start_target:end_target] = target_block
            
            # 3. DAMPING
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

            # NORMALIZATION
            mags = np.abs(self.W_in)
            phases = np.angle(self.W_in)
            mags = np.clip(mags, 0.0, 1.0)
            self.W_in = mags * np.exp(1j * phases)
            
            mags_lat = np.abs(self.W_lat)
            phases_lat = np.angle(self.W_lat)
            mags_lat = np.clip(mags_lat, 0.0, 0.5)
            self.W_lat = mags_lat * np.exp(1j * phases_lat)
            
            # Zero diagonal
            np.fill_diagonal(self.W_lat, 0.0)

        return prediction == label, prediction, total_energy

    def visualize_cortex_ascii(self, digit_idx):
        neuron_idx = digit_idx * self.neurons_per_class
        w_vec = self.W_in[:, neuron_idx]
        w_vec = w_vec[:784]
        mags = np.abs(w_vec)
        
        print(f"\n--- Cortical State (Ch 0) Output {digit_idx} ---")
        try:
            side = 28
            grid_mag = mags.reshape(side, side)
            max_mag = np.max(grid_mag)
            if max_mag == 0: max_mag = 1.0
            for r in range(0, side, 2): 
                line = ""
                for c in range(side):
                    m = grid_mag[r, c]
                    if m < (0.2 * max_mag): line += " " 
                    else: line += "#"
                print(line)
        except: pass
