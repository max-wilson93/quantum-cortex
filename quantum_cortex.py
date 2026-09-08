"""Complex-valued cortical column with local Hebbian learning.

State dynamics
--------------
The column is a leaky resonator. Written out before it was coded, as roadmap
1.2 asks:

    s(0)   = 0
    s(t+1) = leak * s(t)  +  W_in^T x  +  lateral_strength * (W_lat^T s(t))
    s(t+1) <- kerr(s(t+1))                 phase shift by kerr_constant * |s|^2
    s(t+1) <- regulate_energy(s(t+1))      clamp or normalise the L2 norm

`leak` is the fraction of the previous state carried forward. Before Phase 1
the state was *assigned* rather than accumulated (`s = drive + feedback`), so
there was no resonator at all: with `W_lat` dead, `time_steps` of 1 and of 99
gave bit-identical predictions.

Two things had to land together for the loop to mean anything. Accumulation
alone changes nothing the readout can see -- with no lateral mixing the state
is just a geometric series in the drive, and `|state|**2`'s argmax is invariant
to a positive rescale. It is lateral coupling (roadmap 1.1) that mixes evidence
across neurons, and only then does Kerr have somewhere to go: it makes the
phase shift depend on intensity, so coherent groups reinforce through `W_lat`
and incoherent ones cancel.

Reproducing the pre-Phase-1 model
---------------------------------
`ModelConfig.legacy()` restores the original dynamics exactly (`leak=0`,
diagonal `W_lat` init, no input phase, `lateral_strength=1`). The "before"
number for every Phase 1 repair is therefore reproducible from the current
code, not merely recoverable from git history, and
`tests/test_fastpath.py` still checks that configuration against a frozen
transcription of the original loop.
"""

import numpy as np

#: Mechanism switches, all defaulting to on.
MECHANISMS = ("lateral_coupling", "recurrence", "kerr", "phase_input", "energy_clamp")

DEFAULT_CONFIG = {
    # --- physics constants ---
    "learning_rate": 0.09,
    "phase_flexibility": 0.1,
    "lateral_strength": 0.16,
    "input_threshold": 0.7,
    "kerr_constant": 0.2,
    "system_energy": 40.0,
    "time_steps": 4,
    "leak": 0.5,

    # --- mechanism switches (Phase 0.4) ---
    "lateral_coupling": True,
    "recurrence": True,
    "kerr": True,
    "phase_input": True,
    "energy_clamp": True,

    # --- structure choices (Phase 1) ---
    # "offdiagonal": small random complex off-diagonal coupling, diagonal zero
    #                from the start (roadmap 1.1).
    # "diagonal":    the original eye * 0.1, which the training step's
    #                fill_diagonal deletes entirely on the first sample. Kept
    #                so the pre-repair model stays runnable.
    "lateral_init": "offdiagonal",
    "lateral_init_scale": 0.02,

    # "gabor":     use the local Gabor phase carried by complex features
    #              (roadmap 1.3, Option A -- most faithful to the optics).
    # "magnitude": theta = pi * |feature| at unit amplitude (Option B, closer
    #              to a phase-only spatial light modulator).
    # "none":      phase identically zero, as before Phase 1.
    "phase_encoding": "gabor",

    # "matched":      rotate each weight's phase toward the CONJUGATE of the
    #                 input phase, so a matching input sums coherently. This is
    #                 holographic recording: store the conjugate of the
    #                 reference wave. With a zero-phase input it reduces
    #                 exactly to "toward_zero".
    # "toward_zero":  rotate toward phase 0, as before Phase 1. Correct only
    #                 when the input carries no phase.
    "phase_rule": "matched",

    # "clamp":     scale down only when the norm exceeds system_energy.
    # "normalize": always scale to exactly system_energy (roadmap 1.4).
    "energy_mode": "clamp",

    # --- weight initialisation ---
    "init": "uniform",
}

PHASE_ENCODINGS = ("gabor", "magnitude", "none")
ENERGY_MODES = ("clamp", "normalize")
PHASE_RULES = ("matched", "toward_zero")


class QuantumCortex:
    """One cortical column: complex input weights, complex lateral weights.

    Parameters
    ----------
    num_inputs, num_classes, neurons_per_class
        Shape of the column. Outputs are contiguous per-class blocks of
        ``neurons_per_class`` neurons.
    config
        Overrides for :data:`DEFAULT_CONFIG`. Unknown keys raise, so a typo in
        an ablation config fails loudly instead of silently doing nothing.
    seed
        Seeds this column's private generator. Every reported number in this
        repository comes from a seeded run.
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

        if cfg["phase_encoding"] not in PHASE_ENCODINGS:
            raise ValueError(f"phase_encoding must be one of {PHASE_ENCODINGS}")
        if cfg["energy_mode"] not in ENERGY_MODES:
            raise ValueError(f"energy_mode must be one of {ENERGY_MODES}")
        if cfg["phase_rule"] not in PHASE_RULES:
            raise ValueError(f"phase_rule must be one of {PHASE_RULES}")

        self.learning_rate = cfg["learning_rate"]
        self.phase_flexibility = cfg["phase_flexibility"]
        self.input_threshold = cfg["input_threshold"]
        self.kerr_constant = cfg["kerr_constant"]
        self.system_energy = cfg["system_energy"]
        self.time_steps = cfg["time_steps"]
        self.leak = cfg["leak"]
        self.phase_encoding = cfg["phase_encoding"]
        self.energy_mode = cfg["energy_mode"]
        self.phase_rule = cfg["phase_rule"]

        # Scales the lateral feedback term. Before Phase 1.1 this was read from
        # config and never referenced; the lateral update scaled by
        # learning_rate instead and W_lat was identically zero anyway.
        self.lateral_strength = cfg["lateral_strength"]

        for name in MECHANISMS:
            setattr(self, name, bool(cfg[name]))

        self.init_lr = self.learning_rate
        self.init_flex = self.phase_flexibility

        self.rng = np.random.default_rng(seed)
        self._init_weights(cfg["init"], cfg["lateral_init"], cfg["lateral_init_scale"])

    # ------------------------------------------------------------------ setup

    def _init_weights(self, mode, lateral_init, lateral_scale):
        shape = (self.num_inputs, self.num_outputs)
        if mode == "uniform":
            mags = np.full(shape, 0.05)
            phases = np.zeros(shape)
        elif mode == "random":
            mags = self.rng.uniform(0.0, 0.1, size=shape)
            phases = self.rng.uniform(0.0, 2 * np.pi, size=shape)
        else:
            raise ValueError(f"unknown init mode: {mode!r}")
        self.W_in = mags * np.exp(1j * phases)

        n = self.num_outputs
        if lateral_init == "offdiagonal":
            # Roadmap 1.1. The intent behind fill_diagonal(W_lat, 0) is right --
            # a neuron should not drive itself -- but the matrix has to start
            # with something off the diagonal for the Hebbian rule to grow.
            magnitude = self.rng.uniform(0.0, lateral_scale, size=(n, n))
            phase = self.rng.uniform(0.0, 2 * np.pi, size=(n, n))
            self.W_lat = magnitude * np.exp(1j * phase)
            np.fill_diagonal(self.W_lat, 0.0)
        elif lateral_init == "diagonal":
            # The original. A pure diagonal, which the first training step
            # zeroes in full. Retained so the pre-repair model stays runnable.
            self.W_lat = np.eye(n, dtype=complex) * 0.1
        else:
            raise ValueError(f"unknown lateral_init: {lateral_init!r}")

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

        The gate is a hard threshold on feature magnitude, so the input is
        binary and sparse regardless of the phase encoding (roadmap 3.7 turns
        that sparsity into an event-driven readout).

        Phase depends on ``phase_encoding``:

        * ``"gabor"`` -- the phase already carried by complex features. Because
          each Fourier mask is a single-sided wedge and a real image has a
          Hermitian spectrum, every filtered channel is an analytic signal, so
          this is the local Gabor phase: edge position and polarity. Real
          features carry no phase, so this silently reduces to zero unless the
          front-end is asked for ``complex_output=True``.
        * ``"magnitude"`` -- theta = pi * |feature|, unit amplitude. Closer to a
          phase-only spatial light modulator.
        * ``"none"`` -- zero, as before Phase 1.

        The ``phase_input`` ablation switch forces zero phase whatever the
        encoding, so an ablation compares an identical input gate with and
        without phase.
        """
        magnitude = np.abs(feature_vector)
        gate = np.where(magnitude > self.input_threshold, 1.0, 0.0)

        if not self.phase_input or self.phase_encoding == "none":
            return gate.astype(complex)
        if self.phase_encoding == "magnitude":
            return gate * np.exp(1j * np.pi * magnitude)
        phase = np.angle(feature_vector) if np.iscomplexobj(feature_vector) \
            else np.zeros_like(gate)
        return gate * np.exp(1j * phase)

    # ------------------------------------------------------------------ state

    def regulate_energy(self, state_vector):
        """Global energy homeostasis. Two honest options, named for what they are.

        * ``energy_mode="clamp"`` -- a **ceiling clamp**: scale down only when
          the L2 norm exceeds ``system_energy``, otherwise pass through. This
          is what the code has always done. It is not a normalisation (the norm
          is not held constant) and it is not unitary (a rescale is not a
          rotation), whatever the original README said.
        * ``energy_mode="normalize"`` -- a true L2 normalisation: always scale
          to exactly ``system_energy``, putting the state on a fixed sphere.

        Roadmap 1.4 asked for this to be one thing, implemented honestly and
        described accurately. Both are implemented; the ablation table reports
        what each is worth rather than the choice being asserted.
        """
        energy = np.linalg.norm(state_vector)
        if energy == 0:
            return state_vector
        scale = self.system_energy / energy
        if self.energy_mode == "normalize":
            return state_vector * scale
        return state_vector * scale if scale < 1.0 else state_vector

    def _settle(self, input_wave):
        """Run the leaky resonator; see the module docstring for the equation."""
        drive = input_wave @ self.W_in
        state = np.zeros(self.num_outputs, dtype=complex)
        steps = self.time_steps if self.recurrence else 1

        for _ in range(steps):
            if self.lateral_coupling:
                state = self.leak * state + drive + self.lateral_strength * (state @ self.W_lat)
            else:
                state = self.leak * state + drive

            if self.kerr:
                mags = np.abs(state)
                phases = np.angle(state)
                state = mags * np.exp(1j * (phases + self.kerr_constant * mags**2))
            if self.energy_clamp:
                state = self.regulate_energy(state)
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
        """Local Hebbian update. Vectorised; see tests/test_fastpath.py.

        Two rules, both local to the neurons involved:

        * **potentiation** on the target class block -- rotate each active
          weight's phase a fraction ``phase_flexibility`` toward its target,
          then grow its magnitude by ``(1 + learning_rate)``. Under
          ``phase_rule="matched"`` the target is the **conjugate of the input
          phase**, so that a matching input makes ``x_i * w_i`` line up and add
          coherently. That is holographic recording, and it is what makes
          phase-carrying input usable at all: rotating toward zero instead
          (``phase_rule="toward_zero"``, the pre-Phase-1 rule) leaves weight and
          input phases unrelated, the readout sum becomes a random walk of
          magnitude ~sqrt(N) rather than a coherent ~N, and accuracy collapses.
          With a zero-phase input the two rules are identical;
        * **damping** on the wrongly-predicted block, on errors only -- kick the
          phase by uniform noise and shrink the magnitude.

        Magnitudes are then clipped into ``[0, 1]``. Roadmap 3.2 records what
        this multiplicative-growth-plus-clip rule does to the weight
        distribution and replaces it with a self-normalising one.
        """
        npc = self.neurons_per_class
        active = np.nonzero(np.abs(input_wave) > 0.1)[0]
        touched_blocks = []

        if active.size > 0:
            target = np.arange(label * npc, (label + 1) * npc)
            w = self.W_in[np.ix_(active, target)]
            if self.phase_rule == "matched":
                desired = -np.angle(input_wave[active])[:, None]
                # np.angle of the unit phasor wraps the difference into (-pi, pi],
                # so the rotation always takes the short way round.
                delta = np.angle(np.exp(1j * (desired - np.angle(w))))
                w = w * np.exp(1j * self.phase_flexibility * delta)
            else:
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

        if touched_blocks and active.size > 0:
            cols = np.unique(np.concatenate(touched_blocks))
            sub = self.W_in[np.ix_(active, cols)]
            mags = np.clip(np.abs(sub), 0.0, 1.0)
            self.W_in[np.ix_(active, cols)] = mags * np.exp(1j * np.angle(sub))

        if self.lateral_coupling:
            mags_lat = np.clip(np.abs(self.W_lat), 0.0, 0.5)
            self.W_lat = mags_lat * np.exp(1j * np.angle(self.W_lat))
            # A neuron does not drive itself. With the off-diagonal init this
            # removes self-drive; with the legacy diagonal init it deletes the
            # entire matrix, which is the bug roadmap 1.1 repairs.
            np.fill_diagonal(self.W_lat, 0.0)

    # ---------------------------------------------------------------- display

    def visualize_cortex_ascii(self, digit_idx, channel=0):
        """Print the magnitude field of one neuron's Fourier channel."""
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
