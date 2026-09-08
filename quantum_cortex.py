"""Complex-valued cortical column with local Hebbian learning.

Phase 0 note
------------
This module was reworked to be *measurable*, not to be different. Every
mechanism switch defaults to ``True``, and with all switches on the model
computes exactly what it computed before: same update rule, same constants,
same readout. The changes are:

* every source of randomness is drawn from a per-cortex ``numpy.random.Generator``
  instead of the global RNG, so a run is reproducible from a seed;
* each mechanism named in the README has an on/off switch, so ``ablate.py`` can
  ask what it is worth;
* the per-sample update is vectorised. ``tests/test_fastpath.py`` checks it
  against ``tests/reference_cortex.py``, a frozen transcription of the original
  loop, and requires identical predictions.

Nothing here fixes the dead mechanisms. That is Phase 1, and it happens after
the measuring instrument exists.
"""

import numpy as np

#: Mechanism switches, all defaulting to the behaviour the model already had.
MECHANISMS = ("lateral_coupling", "recurrence", "kerr", "phase_input", "energy_clamp")

DEFAULT_CONFIG = {
    # --- physics constants (the "golden" 90.74% run) ---
    "learning_rate": 0.09,
    "phase_flexibility": 0.1,
    "lateral_strength": 0.16,
    "input_threshold": 0.7,
    "kerr_constant": 0.2,
    "system_energy": 40.0,
    "time_steps": 4,
    # --- mechanism switches (Phase 0.4) ---
    "lateral_coupling": True,
    "recurrence": True,
    "kerr": True,
    "phase_input": True,
    "energy_clamp": True,
    # --- weight initialisation ---
    # "uniform": every column starts at magnitude 0.05, phase 0 (as shipped).
    # "random":  magnitudes U[0, 0.1], phases U[0, 2*pi) -- used by bench.py's
    #            untrained-cortex control, which needs weights that are actually
    #            random. Note that the README describes "random" but the model
    #            has always run "uniform"; see tests/test_mechanisms.py.
    "init": "uniform",
}


class QuantumCortex:
    """One cortical column: complex input weights, complex lateral weights.

    Parameters
    ----------
    num_inputs, num_classes, neurons_per_class
        Shape of the column. Outputs are laid out as contiguous per-class
        blocks of ``neurons_per_class`` neurons.
    config
        Overrides for :data:`DEFAULT_CONFIG`. Unknown keys raise, so a typo in
        an ablation config fails loudly instead of silently doing nothing.
    seed
        Seeds this column's private generator. Pass ``None`` only for throwaway
        work; every reported number in this repository comes from a seeded run.
    """

    def __init__(self, num_inputs, num_classes, neurons_per_class, config=None, seed=None):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.neurons_per_class = neurons_per_class
        self.num_outputs = num_classes * neurons_per_class

        cfg = dict(DEFAULT_CONFIG)
        if config:
            unknown = set(config) - set(DEFAULT_CONFIG)
            if unknown:
                raise KeyError(f"unknown config key(s): {sorted(unknown)}")
            cfg.update(config)
        self.config = cfg

        self.learning_rate = cfg["learning_rate"]
        self.phase_flexibility = cfg["phase_flexibility"]
        self.input_threshold = cfg["input_threshold"]
        self.kerr_constant = cfg["kerr_constant"]
        self.system_energy = cfg["system_energy"]
        self.time_steps = cfg["time_steps"]

        # Read from config and stored for compatibility, but never used by any
        # code path. The lateral update below scales by `learning_rate`, not by
        # this. Left visible rather than deleted so the dead parameter is
        # obvious; Phase 1.1 either wires it up or removes it.
        self.lateral_strength = cfg["lateral_strength"]

        for name in MECHANISMS:
            setattr(self, name, bool(cfg[name]))

        # Annealing endpoints.
        self.init_lr = self.learning_rate
        self.init_flex = self.phase_flexibility

        self.rng = np.random.default_rng(seed)
        self._init_weights(cfg["init"])

    # ------------------------------------------------------------------ setup

    def _init_weights(self, mode):
        shape = (self.num_inputs, self.num_outputs)
        if mode == "uniform":
            # As shipped: every neuron identical, phase exactly zero. The
            # README's "theta ~ U[0, 2*pi]" has never been what the code does.
            mags = np.full(shape, 0.05)
            phases = np.zeros(shape)
        elif mode == "random":
            mags = self.rng.uniform(0.0, 0.1, size=shape)
            phases = self.rng.uniform(0.0, 2 * np.pi, size=shape)
        else:
            raise ValueError(f"unknown init mode: {mode!r}")
        self.W_in = mags * np.exp(1j * phases)

        # Pure diagonal, as shipped. The first training step zeroes the
        # diagonal, which zeroes the entire matrix -- that is the bug Phase 1.1
        # repairs, and it is left in place here so Phase 0 can measure it.
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * 0.1
        if not self.lateral_coupling:
            self.W_lat[:] = 0.0

    def decay_learning_rate(self, progress):
        """Linearly anneal plasticity. ``progress`` runs 0 -> 1 over training."""
        decay = 1.0 - (progress * 0.9)
        self.learning_rate = self.init_lr * decay
        self.phase_flexibility = self.init_flex * decay

    # ------------------------------------------------------------------ front

    def encode_input(self, feature_vector):
        """Gate the feature vector and attach a phase.

        The gate is a hard threshold on feature magnitude, which is why the
        input is binary and sparse (see roadmap 3.7).

        The phase is taken from the features when they are complex and is zero
        when they are real. Today ``FourierOptics.apply`` calls ``np.abs()``
        before returning, so the features are always real and the phase is
        always zero -- the ``phase_input`` switch therefore does nothing yet,
        and ``ablate.py`` will report exactly that. Phase 1.3 makes the
        front-end pass phase through, at which point this code path carries it
        without further change.
        """
        magnitude = np.abs(feature_vector)
        gate = np.where(magnitude > self.input_threshold, 1.0, 0.0)
        if self.phase_input and np.iscomplexobj(feature_vector):
            phase = np.angle(feature_vector)
        else:
            phase = np.zeros_like(gate)
        return gate * np.exp(1j * phase)

    # ------------------------------------------------------------------ state

    def clamp_energy(self, state_vector):
        """Scale the state down if its L2 norm exceeds ``system_energy``.

        Named honestly: this is a **ceiling clamp, not a normalisation**. It
        only ever scales down (``if scale < 1.0``), so a state below the
        ceiling passes through untouched and the norm is not held constant. It
        is also not unitary -- a uniform rescale is not a rotation. The README
        calls it "Unitary L2 Normalization"; that description is wrong on both
        counts and Phase 4.5 corrects it.

        Roadmap 1.4 decides whether this becomes a true normalisation or stays
        a clamp. Until that decision is made, the behaviour is unchanged and
        only the name and this docstring tell the truth about it.
        """
        energy = np.linalg.norm(state_vector)
        if energy > 0:
            scale = self.system_energy / energy
            if scale < 1.0:
                state_vector = state_vector * scale
        return state_vector

    def _settle(self, input_wave):
        """Run the recurrent loop and return the final complex state.

        As written, ``state`` is *assigned* rather than accumulated, so each
        timestep discards the previous one except through ``feedback``. With
        ``W_lat`` zero (which it is after the first training sample) feedback
        vanishes and the loop reduces to a single feedforward pass. Phase 1.2
        replaces this with an explicit difference equation.
        """
        state = np.zeros(self.num_outputs, dtype=complex)
        steps = self.time_steps if self.recurrence else 1
        for _ in range(steps):
            feedforward = input_wave @ self.W_in
            if self.recurrence and self.lateral_coupling:
                state = feedforward + state @ self.W_lat
            else:
                state = feedforward
            if self.kerr:
                mags = np.abs(state)
                phases = np.angle(state)
                state = mags * np.exp(1j * (phases + self.kerr_constant * mags**2))
            if self.energy_clamp:
                state = self.clamp_energy(state)
        return state

    def readout(self, state):
        """Per-class energy: sum of ``|state|**2`` over each class block."""
        energies = np.abs(state) ** 2
        class_energies = energies.reshape(self.num_classes, self.neurons_per_class).sum(axis=1)
        return energies, class_energies

    # --------------------------------------------------------------- learning

    def process_image(self, feature_vector, label, train=True):
        """Classify one sample, optionally applying the local update.

        Returns ``(correct, prediction, total_energy)``.
        """
        feature_vector = np.asarray(feature_vector).ravel()
        input_wave = self.encode_input(feature_vector)

        state = self._settle(input_wave)
        energies, class_energies = self.readout(state)
        prediction = int(np.argmax(class_energies))

        if train:
            self._update(input_wave, label, prediction)

        return prediction == label, prediction, float(np.sum(energies))

    def _update(self, input_wave, label, prediction):
        """Local Hebbian update. Vectorised; see module docstring for the check.

        Two rules, both local to the neurons involved:

        * **potentiation** on the target class block -- rotate each active
          weight's phase a fraction ``phase_flexibility`` toward zero, then
          grow its magnitude by ``(1 + learning_rate)``;
        * **damping** on the wrongly-predicted block, on errors only -- kick the
          phase by uniform noise and shrink the magnitude.

        Magnitudes are then clipped into ``[0, 1]``. Roadmap 3.2 records what
        this multiplicative-growth-plus-clip rule does to the weight
        distribution over 60k samples, and replaces it.
        """
        npc = self.neurons_per_class
        active = np.nonzero(np.abs(input_wave) > 0.1)[0]
        touched_blocks = []

        if active.size > 0:
            target = np.arange(label * npc, (label + 1) * npc)
            w = self.W_in[np.ix_(active, target)]
            w = w * np.exp(-1j * self.phase_flexibility * np.angle(w))
            w = w * (1.0 + self.learning_rate)
            self.W_in[np.ix_(active, target)] = w
            touched_blocks.append(target)

            if self.lateral_coupling:
                block = slice(label * npc, (label + 1) * npc)
                self.W_lat[block, block] *= (1.0 + self.learning_rate)

        if prediction != label:
            wrong = np.arange(prediction * npc, (prediction + 1) * npc)
            # Drawn as (npc, k) so the flat stream matches npc successive
            # k-length draws, i.e. the original per-neuron loop.
            noise = self.rng.uniform(-1.0, 1.0, size=(npc, active.size)).T
            w = self.W_in[np.ix_(active, wrong)]
            w = w * np.exp(1j * self.phase_flexibility * noise)
            w = w * (1.0 - self.learning_rate)
            self.W_in[np.ix_(active, wrong)] = w
            touched_blocks.append(wrong)

        # Clip only what changed. The original clipped all num_inputs x
        # num_outputs weights every sample; untouched entries are already
        # inside the bound, so restricting the clip is a pure speedup.
        if touched_blocks and active.size > 0:
            cols = np.unique(np.concatenate(touched_blocks))
            sub = self.W_in[np.ix_(active, cols)]
            mags = np.clip(np.abs(sub), 0.0, 1.0)
            self.W_in[np.ix_(active, cols)] = mags * np.exp(1j * np.angle(sub))

        if self.lateral_coupling:
            mags_lat = np.clip(np.abs(self.W_lat), 0.0, 0.5)
            self.W_lat = mags_lat * np.exp(1j * np.angle(self.W_lat))
            # A neuron should not drive itself. But W_lat starts as a pure
            # diagonal, so this deletes the whole matrix on the first sample.
            np.fill_diagonal(self.W_lat, 0.0)

    # ---------------------------------------------------------------- display

    def visualize_cortex_ascii(self, digit_idx, channel=0):
        """Print the magnitude field of one neuron's first Fourier channel."""
        side = 28
        neuron_idx = digit_idx * self.neurons_per_class
        offset = channel * side * side
        w_vec = self.W_in[offset:offset + side * side, neuron_idx]
        if w_vec.size != side * side:
            print(f"[skip] channel {channel} does not cover {side}x{side} inputs "
                  f"(got {w_vec.size} of {self.num_inputs})")
            return

        grid = np.abs(w_vec).reshape(side, side)
        peak = grid.max() or 1.0
        print(f"\n--- Cortical State (Ch {channel}) Output {digit_idx} ---")
        for r in range(0, side, 2):
            print("".join(" " if grid[r, c] < 0.2 * peak else "#" for c in range(side)))
