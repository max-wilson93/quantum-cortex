"""Frozen transcription of the original QuantumCortex, for equivalence testing.

This is the model exactly as it stood at commit ``babdbce``, before Phase 0
touched it. It is deliberately **not** a subclass of the production class and
deliberately not tidied up: its whole value is that it cannot drift when
``quantum_cortex.py`` changes.

One single deviation from the original, required to compare anything at all:
``np.random.uniform`` is replaced by a passed-in ``numpy.random.Generator``, so
the reference and the production class can be driven from the same stream. The
draw shapes and their order are preserved exactly.

Used by ``tests/test_fastpath.py`` to prove that Phase 0's vectorised update is
the same computation, not a new one.
"""

import numpy as np


class ReferenceCortex:
    def __init__(self, num_inputs, num_classes, neurons_per_class, config=None, seed=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_classes * neurons_per_class
        self.neurons_per_class = neurons_per_class
        self.num_classes = num_classes

        cfg = config or {}
        self.learning_rate = cfg.get("learning_rate", 0.09)
        self.phase_flexibility = cfg.get("phase_flexibility", 0.1)
        self.lateral_strength = cfg.get("lateral_strength", 0.16)
        self.input_threshold = cfg.get("input_threshold", 0.7)
        self.kerr_constant = cfg.get("kerr_constant", 0.2)
        self.system_energy = cfg.get("system_energy", 40.0)

        self.init_lr = self.learning_rate
        self.init_flex = self.phase_flexibility
        self.time_steps = cfg.get("time_steps", 4)

        init_mag = np.ones((num_inputs, self.num_outputs)) * 0.05
        self.W_in = init_mag * np.exp(1j * np.zeros((num_inputs, self.num_outputs)))
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1

        self.rng = np.random.default_rng(seed)

    def decay_learning_rate(self, progress):
        decay_factor = 1.0 - (progress * 0.9)
        self.learning_rate = self.init_lr * decay_factor
        self.phase_flexibility = self.init_flex * decay_factor

    def get_phasic_input(self, feature_vector):
        mag = np.where(feature_vector > self.input_threshold, 1.0, 0.0)
        return mag * np.exp(1j * 0)

    def normalize_state(self, state_vector):
        current_energy = np.linalg.norm(state_vector)
        if current_energy > 0:
            scale = self.system_energy / current_energy
            if scale < 1.0:
                state_vector *= scale
        return state_vector

    def process_image(self, feature_vector, label, train=True):
        if feature_vector.ndim > 1:
            feature_vector = feature_vector.flatten()
        input_wave = self.get_phasic_input(feature_vector)

        cortex_state = np.zeros(self.num_outputs, dtype=complex)

        for t in range(self.time_steps):
            feedforward = np.dot(input_wave, self.W_in)
            feedback = np.dot(cortex_state, self.W_lat)
            cortex_state = feedforward + feedback

            mags = np.abs(cortex_state)
            phases = np.angle(cortex_state)
            kerr_shift = self.kerr_constant * (mags ** 2)
            cortex_state = mags * np.exp(1j * (phases + kerr_shift))

            cortex_state = self.normalize_state(cortex_state)

        energies = np.abs(cortex_state) ** 2
        total_energy = np.sum(energies)

        class_energies = np.zeros(10)
        for c in range(10):
            start = c * self.neurons_per_class
            end = start + self.neurons_per_class
            class_energies[c] = np.sum(energies[start:end])

        prediction = np.argmax(class_energies)

        if train:
            start_target = label * self.neurons_per_class
            end_target = start_target + self.neurons_per_class
            active_inputs = np.where(np.abs(input_wave) > 0.1)[0]

            if len(active_inputs) > 0:
                for n in range(start_target, end_target):
                    w_sub = self.W_in[active_inputs, n]
                    w_phase = np.angle(w_sub)
                    rot = np.exp(-1j * self.phase_flexibility * w_phase)
                    w_sub = w_sub * rot
                    w_sub = w_sub * (1.0 + self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

                target_block = self.W_lat[start_target:end_target, start_target:end_target]
                target_block = target_block * (1.0 + self.learning_rate)
                self.W_lat[start_target:end_target, start_target:end_target] = target_block

            if prediction != label:
                start_wrong = prediction * self.neurons_per_class
                end_wrong = start_wrong + self.neurons_per_class
                for n in range(start_wrong, end_wrong):
                    w_sub = self.W_in[active_inputs, n]
                    noise = self.rng.uniform(-1.0, 1.0, size=len(active_inputs))
                    dec = np.exp(1j * self.phase_flexibility * noise)
                    w_sub = w_sub * dec
                    w_sub = w_sub * (1.0 - self.learning_rate)
                    self.W_in[active_inputs, n] = w_sub

            mags = np.abs(self.W_in)
            phases = np.angle(self.W_in)
            mags = np.clip(mags, 0.0, 1.0)
            self.W_in = mags * np.exp(1j * phases)

            mags_lat = np.abs(self.W_lat)
            phases_lat = np.angle(self.W_lat)
            mags_lat = np.clip(mags_lat, 0.0, 0.5)
            self.W_lat = mags_lat * np.exp(1j * phases_lat)

            np.fill_diagonal(self.W_lat, 0.0)

        return prediction == label, prediction, total_energy
