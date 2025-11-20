import numpy as np
import cmath

class QuantumCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes
        
        # --- PHYSICS CONFIG ---
        self.learning_rate = 0.025
        self.phase_flexibility = 0.125
        self.input_threshold = 0.35
        self.kerr_constant = 0.5
        
        # --- CORTEX CONFIG ---
        self.time_steps = 3 
        self.lateral_strength = 0.1 # Connection strength between neurons
        
        # --- HOMEOSTASIS (Target Energy) ---
        # We want the total energy of the system to hover around this value.
        # If it goes higher, we mathematically shrink the weights.
        self.target_energy = 5.0 
        self.scaling_factor = 1.0 # Monitor this to see stability
        
        # Weights
        # Feedforward (Input -> Cortex)
        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.01
        self.W_in = init_mag * np.exp(1j * np.zeros((num_inputs, self.num_outputs)))
        
        # Lateral (Cortex <-> Cortex)
        # Identity matrix (Self-excitation) + small random noise
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def renormalize_system(self, current_energy):
        """
        SYNAPTIC SCALING:
        If the brain is too loud (Energy > Target), we shrink ALL weights proportionally.
        This preserves the *Relative Phase* (The Information) while fixing the *Instability* (The Heat).
        """
        if current_energy > self.target_energy:
            # Calculate how much we need to shrink
            # Energy is proportional to Amplitude^2, so we scale amplitude by sqrt(ratio)
            ratio = self.target_energy / (current_energy + 1e-9)
            scale = np.sqrt(ratio)
            
            # Smooth update (Don't snap instantly, drift towards stability)
            # This acts like a dampener
            self.scaling_factor = 0.9 * self.scaling_factor + 0.1 * scale
            
            # Apply scaling to weights
            # We only scale down, never up (prevent explosion)
            if self.scaling_factor < 1.0:
                self.W_in *= self.scaling_factor
                self.W_lat *= self.scaling_factor
                # Reset factor so we don't double-apply
                self.scaling_factor = 1.0

    def process_image(self, feature_vector, label, train=True):
        if feature_vector.ndim > 1: feature_vector = feature_vector.flatten()
        input_wave = self.get_phasic_input(feature_vector)
        
        cortex_state = np.zeros(self.num_outputs, dtype=complex)
        
        # --- RECURRENT LOOP ---
        for t in range(self.time_steps):
            # 1. Input
            feedforward = np.dot(input_wave, self.W_in)
            
            # 2. Lateral
            feedback = np.dot(cortex_state, self.W_lat)
            
            # 3. Integrate
            cortex_state = feedforward + feedback
            
            # 4. Kerr Nonlinearity
            mags = np.abs(cortex_state)
            phases = np.angle(cortex_state)
            kerr_shift = self.kerr_constant * (mags ** 2)
            cortex_state = mags * np.exp(1j * (phases + kerr_shift))
            
            # 5. Instant Normalization (Refractory)
            # Prevents run-away values inside the loop
            max_val = np.max(mags)
            if max_val > 1.0:
                cortex_state /= max_val

        # --- READOUT ---
        energies = np.abs(cortex_state) ** 2
        
        class_energies = np.zeros(10)
        for c in range(10):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_energies[c] = np.sum(energies[start:end])
            
        prediction = np.argmax(class_energies)
        
        # --- LEARNING ---
        if train:
            # Renormalize Weights based on Total Energy produced
            total_output_energy = np.sum(energies)
            self.renormalize_system(total_output_energy)
            
            start_target = label * self.neurons_per_class
            end_target = start_target + self.neurons_per_class
            
            active_inputs = np.where(np.abs(input_wave) > 0.1)[0]
            
            if len(active_inputs) > 0:
                # A. Resonance
                for n in range(start_target, end_target):
                    w_sub = self.W_in[active_inputs, n]
                    w_phase = np.angle(w_sub)
                    # Rotate to 0
                    rot = np.exp(-1j * self.phase_flexibility * w_phase)
                    w_sub *= rot
                    w_sub *= (1.0 + self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

                # B. Lateral Hebbian (Cluster the Class)
                # Boost connections between neurons of the target class
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

            # Hard Clip (Safety Net)
            mags = np.abs(self.W_in)
            phases = np.angle(self.W_in)
            mags = np.clip(mags, 0.0, 1.0)
            self.W_in = mags * np.exp(1j * phases)
            
            mags_lat = np.abs(self.W_lat)
            phases_lat = np.angle(self.W_lat)
            mags_lat = np.clip(mags_lat, 0.0, 0.5) 
            self.W_lat = mags_lat * np.exp(1j * phases_lat)

        return prediction == label, prediction

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