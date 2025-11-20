import numpy as np
import random

class QuantumEntanglementSNN:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # ARCHITECTURE
        self.real_inputs = num_inputs
        self.total_input_channels = num_inputs * 2 
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # INITIALIZATION
        self.w_in_hidden = np.random.uniform(0.0, 1.0, (self.total_input_channels, num_hidden))
        self.w_hidden_out = np.random.uniform(0.0, 1.0, (num_hidden, num_outputs))
        
        # FIX 1: ENERGY REBALANCING
        # Hidden: Reduced 8.0 -> 4.0 (Prevent Seizure during alignment)
        # Output: Increased 4.0 -> 12.0 (Ensure sparse hidden activity drives output)
        self.norm_factor_hidden = 4.0 
        self.norm_factor_out = 12.0 
        self.normalize_weights()

        # PHYSICS
        self.v_rest = 0.0
        self.v_base_thresh = 1
        self.decay = .95

        # STATE
        self.v_hidden = np.zeros(num_hidden)
        self.v_out = np.zeros(num_outputs)
        
        # THRESHOLDS (Homeostasis)
        self.thresh_hidden = np.ones(num_hidden) * self.v_base_thresh
        self.thresh_out = np.ones(num_outputs) * self.v_base_thresh
        
        self.t_input = np.zeros(self.total_input_channels) - 1000
        self.t_hidden = np.zeros(num_hidden) - 1000
        self.t_out = np.zeros(num_outputs) - 1000

        # MEMORY
        self.episodic_memory = []
        self.max_memory_size = 300 
        self.learning_rate = 0.05

    def normalize_weights(self):
        # Input -> Hidden
        col_sums = np.sum(self.w_in_hidden, axis=0)
        col_sums[col_sums == 0] = 1.0
        self.w_in_hidden *= (self.norm_factor_hidden / col_sums)
        
        # Hidden -> Output
        col_sums_out = np.sum(self.w_hidden_out, axis=0)
        col_sums_out[col_sums_out == 0] = 1.0
        self.w_hidden_out *= (self.norm_factor_out / col_sums_out)

    def get_polarity_input(self, image_data):
        on_probs = image_data * 0.25 
        off_probs = (1.0 - image_data) * 0.02 
        return np.concatenate([on_probs, off_probs])

    def vectorized_update(self, t, last_spike_times, fired_indices, weights, reward, lr):
        if len(fired_indices) == 0: return weights
        dt = t - last_spike_times
        relevant_mask = (dt < 40) & (dt > 0)
        if not np.any(relevant_mask): return weights
        
        dts = dt[relevant_mask]
        strength = (1.0 / (np.log(dts) + 1.0)) 
        dw_vec = strength * reward * lr

        rows = np.where(relevant_mask)[0]
        cols = fired_indices
        weights[np.ix_(rows, cols)] += dw_vec[:, np.newaxis]
        
        return weights

    def process_image_logic(self, image_data, label, train=True):
        steps = 40
        input_probs = self.get_polarity_input(image_data)
        
        self.v_hidden.fill(self.v_rest)
        self.v_out.fill(self.v_rest)
        output_spikes_count = np.zeros(self.num_outputs)
        
        max_v_h = 0.0
        max_v_o = 0.0

        for t in range(steps):
            # 1. RETINA
            input_spikes = np.random.rand(self.total_input_channels) < input_probs
            input_indices = np.where(input_spikes)[0]
            self.t_input[input_indices] = t

            # 2. HIDDEN LAYER
            if len(input_indices) > 0:
                hidden_current = np.sum(self.w_in_hidden[input_indices, :], axis=0)
            else:
                hidden_current = np.zeros(self.num_hidden)

            # Noise injection
            noise = np.random.normal(0, 0.05, self.num_hidden)
            
            self.v_hidden = (self.v_hidden * self.decay) + hidden_current + noise
            max_v_h = max(max_v_h, np.max(self.v_hidden))
            
            firing_hidden = np.where(self.v_hidden >= self.thresh_hidden)[0]
            
            if len(firing_hidden) > 0:
                self.v_hidden[firing_hidden] = self.v_rest
                self.t_hidden[firing_hidden] = t
                
                # FIX 2: CAP REFRACTORY GROWTH
                # Allow threshold to rise, but cap it at 3.0 to prevent "death by adaptation"
                self.thresh_hidden[firing_hidden] += 0.3
                self.thresh_hidden = np.minimum(self.thresh_hidden, 3.0)
                
                # Lateral Inhibition
                mask = np.ones(self.num_hidden, dtype=bool)
                mask[firing_hidden] = False
                self.v_hidden[mask] -= 0.5 

                if train:
                    self.w_in_hidden = self.vectorized_update(
                        t, self.t_input, firing_hidden, self.w_in_hidden, 1.0, 0.05
                    )

            # 3. OUTPUT LAYER
            if len(firing_hidden) > 0:
                out_current = np.sum(self.w_hidden_out[firing_hidden, :], axis=0)
            else:
                out_current = np.zeros(self.num_outputs)
                
            self.v_out = (self.v_out * self.decay) + out_current
            max_v_o = max(max_v_o, np.max(self.v_out))
            
            firing_out = np.where(self.v_out >= self.thresh_out)[0]
            output_spikes_count[firing_out] += 1
            self.v_out[firing_out] = self.v_rest
            self.t_out[firing_out] = t
            
            if len(firing_out) > 0:
                self.thresh_out[firing_out] += 0.5
                
                mask = np.ones(self.num_outputs, dtype=bool)
                mask[firing_out] = False
                self.v_out[mask] -= 2.0

                if train:
                    correct = firing_out[firing_out == label]
                    wrong = firing_out[firing_out != label]
                    
                    if len(correct) > 0:
                        self.w_hidden_out = self.vectorized_update(
                            t, self.t_hidden, correct, self.w_hidden_out, 1.0, 0.08
                        )
                        # Decoherence
                        recent_hidden = (t - self.t_hidden < 20) & (t - self.t_hidden > 0)
                        if np.any(recent_hidden):
                            r = np.where(recent_hidden)[0]
                            c = np.array([x for x in range(self.num_outputs) if x != label])
                            self.w_hidden_out[np.ix_(r, c)] -= 0.01

                    if len(wrong) > 0:
                        self.w_hidden_out = self.vectorized_update(
                            t, self.t_hidden, wrong, self.w_hidden_out, -0.5, 0.05
                        )

        # --- HOMEOSTASIS ---
        # Decay thresholds
        self.thresh_hidden = 0.99 * self.thresh_hidden + 0.01 * 1.0
        self.thresh_out = 0.99 * self.thresh_out + 0.01 * 1.0
        
        # Recovery for silent neurons
        self.thresh_hidden -= 0.002
        self.thresh_out -= 0.002
        self.thresh_hidden = np.maximum(self.thresh_hidden, 0.5)
        self.thresh_out = np.maximum(self.thresh_out, 0.5)

        if train:
            self.normalize_weights()

        prediction = np.argmax(output_spikes_count)
        if np.sum(output_spikes_count) == 0: prediction = -1
        
        return prediction == label, prediction, max_v_h, max_v_o

    def remember(self, image, label):
        self.episodic_memory.append((image, label))
        if len(self.episodic_memory) > self.max_memory_size:
            self.episodic_memory.pop(0) 

    def sleep(self):
        if not self.episodic_memory: return
        print(f"    [State] Thresh H: {np.mean(self.thresh_hidden):.2f} | V_Max H: {np.max(self.v_hidden):.2f}")
        print(f"    [Sleep] Consolidating {len(self.episodic_memory)} memories...", end="")
        
        original_lr = self.learning_rate
        self.learning_rate = 0.10
        random.shuffle(self.episodic_memory)
        for img, lbl in self.episodic_memory:
            self.process_image_logic(img, lbl, train=True)
        self.learning_rate = original_lr
        print(" Done.")

    def process_image(self, image_data, label, train=True):
        is_correct, pred, vh, vo = self.process_image_logic(image_data, label, train)
        if train:
            self.remember(image_data, label)
        return is_correct, pred, vh, vo

    def visualize_weights_ascii(self, digit_idx):
        w = self.w_hidden_out[:, digit_idx]
        print(f"\n--- Interference Pattern for Output {digit_idx} ---")
        try:
            side = int(np.sqrt(self.num_hidden))
            grid = w[:side*side].reshape(side, side)
            w_min, w_max = np.min(grid), np.max(grid)
            w_range = w_max - w_min
            if w_range == 0: w_range = 1.0
            chars = " .:-=+*#%@"
            for r in range(side): 
                line = ""
                for c in range(side):
                    val = grid[r, c]
                    norm_val = (val - w_min) / w_range
                    char_idx = int(norm_val * (len(chars) - 1))
                    line += chars[char_idx] + " "
                print(line)
            print(f"    (Weight Range: {w_min:.4f} to {w_max:.4f})")
        except Exception as e:
            print(f"Vis Error: {e}")