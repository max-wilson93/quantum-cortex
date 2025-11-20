import numpy as np
import random

class QuantumEntanglementSNN:
    def __init__(self, num_inputs, num_classes, neurons_per_class):
        # ARCHITECTURE: Population Coding
        self.num_inputs = num_inputs
        self.num_classes = num_classes # 10 digits
        self.neurons_per_class = neurons_per_class # e.g., 5 neurons per digit
        self.num_outputs = num_classes * neurons_per_class
        
        # INITIALIZATION
        # We initialize with slightly more variance to ensure the 5 neurons 
        # per class start different from each other.
        self.weights = np.random.uniform(0.05, 0.2, (num_inputs, self.num_outputs))
        
        # PHYSICS (Tuned for Precision)
        self.v_rest = 0.0
        self.v_base_thresh = 1.0 
        self.decay = 0.80 # Fast decay = coincidences must be precise
        
        # HOMEOSTASIS
        self.thresh_adaptation = np.zeros(self.num_outputs)
        self.thresh_decay = 0.99 
        self.adaptation_penalty = 0.1 # Stronger penalty to force rotation among the 5 neurons

        # MEMORY
        self.episodic_memory = []
        self.max_memory_size = 500 
        self.learning_rate = 0.02 # Lower rate = finer features

        # STATE
        self.voltages = np.zeros(self.num_outputs)
        self.last_spike_times_pre = np.zeros(num_inputs) - 1000 
        self.last_spike_times_post = np.zeros(self.num_outputs) - 1000

    def get_contrast_input(self, image_data):
        """
        Refinement 1: Sigmoid Contrast Enhancement
        Instead of linear scaling, we squash noise to 0 and boost ink to 1.
        """
        # Center around 0.5, steepness 10
        x = (image_data - 0.5) * 10
        sigmoid = 1 / (1 + np.exp(-x))
        
        # Scale to probability (Max 25% chance to spike per step)
        return sigmoid * 0.25

    def logarithmic_weight_update(self, pre_idx, post_idx, current_time, reward):
        t_pre = self.last_spike_times_pre[pre_idx]
        delta_t = current_time - t_pre
        
        # Refinement 3: Tighter Window (20ms)
        if delta_t > 0 and delta_t < 20: 
            # Quantum Entanglement Rule
            entanglement_strength = 1.0 / (np.log(delta_t + 1.0) + 0.1)
            d_w = self.learning_rate * entanglement_strength * reward
            self.weights[pre_idx, post_idx] += d_w
            self.weights[pre_idx, post_idx] = np.clip(self.weights[pre_idx, post_idx], 0.0, 1.0)

    def process_image_logic(self, image_data, label, train=True):
        simulation_steps = 40 
        spike_probs = self.get_contrast_input(image_data)
        
        self.voltages.fill(self.v_rest)
        output_spikes = np.zeros(self.num_outputs)
        self.thresh_adaptation *= self.thresh_decay

        for t in range(simulation_steps):
            pre_spikes = np.random.rand(self.num_inputs) < spike_probs
            pre_spike_indices = np.where(pre_spikes)[0]
            self.last_spike_times_pre[pre_spike_indices] = t
            
            input_current = np.sum(self.weights[pre_spike_indices, :], axis=0)
            self.voltages = (self.voltages * self.decay) + input_current
            
            effective_thresh = self.v_base_thresh + self.thresh_adaptation
            firing_neurons = np.where(self.voltages >= effective_thresh)[0]
            
            if len(firing_neurons) > 0:
                output_spikes[firing_neurons] += 1
                self.voltages[firing_neurons] = self.v_rest 
                self.thresh_adaptation[firing_neurons] += self.adaptation_penalty
                
                if train:
                    for post_idx in firing_neurons:
                        # Determine which Digit this neuron belongs to
                        neuron_class = post_idx // self.neurons_per_class
                        
                        # Refinement 2: Population Reward
                        # If this neuron belongs to the correct CLASS, reward it.
                        if neuron_class == label:
                            reward = 1.0 
                        else:
                            reward = -0.2 # Softer punishment (let other neurons live)

                        recent_inputs = np.where((t - self.last_spike_times_pre) < 20)[0]
                        for pre_idx in recent_inputs:
                            self.logarithmic_weight_update(pre_idx, post_idx, t, reward)
                            
                self.last_spike_times_post[firing_neurons] = t
                
                # Global Lateral Inhibition (Winner-Take-All)
                mask = np.ones(self.num_outputs, dtype=bool)
                mask[firing_neurons] = False
                self.voltages[mask] -= 2.0 

        # --- PREDICTION LOGIC (Population Voting) ---
        # We sum the spikes for all neurons belonging to Class 0, Class 1, etc.
        class_votes = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_votes[c] = np.sum(output_spikes[start:end])

        prediction = np.argmax(class_votes)
        
        # If silence, return -1
        if np.sum(class_votes) == 0: prediction = -1
            
        return prediction == label, prediction

    def remember(self, image, label):
        self.episodic_memory.append((image, label))
        if len(self.episodic_memory) > self.max_memory_size:
            self.episodic_memory.pop(0) 

    def sleep(self):
        if not self.episodic_memory: return
        print(f"    [Sleep] Consolidating {len(self.episodic_memory)} memories...", end="")
        original_lr = self.learning_rate
        self.learning_rate = 0.05 
        random.shuffle(self.episodic_memory)
        for img, lbl in self.episodic_memory:
            self.process_image_logic(img, lbl, train=True)
        self.learning_rate = original_lr
        print(" Done.")

    def process_image(self, image_data, label, train=True):
        is_correct, pred = self.process_image_logic(image_data, label, train)
        if train:
            self.remember(image_data, label)
        return is_correct, pred