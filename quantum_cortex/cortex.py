"""The quantum-holographic cortex: complex-valued, online, one-shot.

Neurons are resonators rather than switches. Each carries a complex state
``psi = A e^(i theta)``, dendritic integration is wave interference rather than
scalar summation, non-linearity comes from the optical Kerr effect (a strong
signal twists its own phase, self-focusing into a stable soliton while noise
washes out destructively), and a unitary L2 clamp holds the column's total
energy constant so it cannot run away into seizure.

Learning is local and Hebbian on phase. There is no backpropagation, no global
error gradient, and no epoch: every sample is presented once, updates the
weights immediately, and is never seen again.

What changed from the original
------------------------------
The original was written against MNIST and hardcoded to it in ways that made it
unusable elsewhere. Each fix below is behaviour-preserving at the defaults, so
the validated MNIST result still reproduces:

* the readout pooled ``range(10)`` regardless of ``num_classes``, so any other
  problem silently scored against the wrong number of classes;
* ``lateral_strength`` was accepted from the config and never read -- ``W_lat``
  was seeded with a hardcoded ``0.1``. **The published 90.74% run therefore had
  an effective lateral strength of 0.1, not the 0.16 its config records.** The
  default here is 0.1 so that result reproduces; the knob is now real;
* the damping branch drew from the global numpy RNG, so no run was
  reproducible. The engine consuming this guarantees byte-identical repeat
  runs, so every draw now comes from an owned, seeded ``Generator``;
* there was no way to persist a trained cortex, which makes online learning
  pointless -- the whole value is that it keeps learning across sessions;
* the input gate binarised at a threshold, mapping a feature at 0.71 and one at
  6.0 to the same ``1+0j``. See :mod:`quantum_cortex.encoders`;
* the readout returned a bare label. See :mod:`quantum_cortex.readout`.

One measured finding is worth carrying here rather than leaving in a benchmark
file: **the three-cortex "Trinity" ensemble contributes nothing.** As shipped,
all three cortices initialise identically -- there is no randomness in
``__init__`` at all, contrary to the architecture description -- and across
2000 held-out samples they never once disagreed. Initialising with random
phases, which is what the design calls for, produced 14 disagreements and no
accuracy gain, because the phase-Hebbian rule rotates every active weight
toward zero and walks all three into the same attractor whatever they started
from. :class:`~quantum_cortex.ensemble.Ensemble` still exists and is honest
about what it does and does not buy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quantum_cortex.encoders import PhasicEncoding, to_phasic
from quantum_cortex.readout import Prediction

__all__ = ["QuantumCortex", "GOLDEN_CONFIG"]

#: The physics that produced 90.74% on MNIST, found by Monte Carlo search over
#: the parameter space. ``lateral_strength`` is 0.1 rather than the 0.16 the
#: historical config records, because 0.1 is what the code actually applied --
#: see the module docstring.
GOLDEN_CONFIG: dict[str, float] = {
    "learning_rate": 0.09,
    "phase_flexibility": 0.10,
    "lateral_strength": 0.10,
    "input_threshold": 0.70,
    "kerr_constant": 0.20,
    "system_energy": 40.0,
}

_FORMAT_VERSION = 1


class QuantumCortex:
    """One cortical column. Complex weights, resonant integration, phase Hebbian.

    Args:
        num_inputs: Length of the feature vector, i.e. the encoder's
            ``n_features``.
        num_classes: How many classes to pool the readout into. Honoured --
            the original ignored this and always pooled ten.
        neurons_per_class: Output neurons allocated to each class. More
            neurons let one class hold several distinct holograms.
        config: Physics overrides. Anything omitted falls back to
            :data:`GOLDEN_CONFIG`.
        seed: Seeds this cortex's own ``Generator``. Two cortices with the same
            seed, fed the same samples in the same order, are identical.
        encoding: How features become a complex wave. Defaults to the
            historical binary gate; continuous features want something else.
        phase_init: ``"zero"`` reproduces the published runs. ``"random"``
            draws initial phases from U[0, 2pi), which is what the architecture
            description calls for -- measured as near-inert, see the module
            docstring.
        balance_classes: Scale plasticity by inverse class frequency. Off by
            default because it changes the validated numerics; needed for
            skewed problems, where a ~12% positive rate otherwise lets the
            majority class dominate the damping step.
    """

    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        neurons_per_class: int = 5,
        config: dict[str, float] | None = None,
        *,
        seed: int | None = None,
        encoding: PhasicEncoding = PhasicEncoding.BINARY,
        phase_init: Literal["zero", "random"] = "zero",
        balance_classes: bool = False,
        time_steps: int = 4,
    ) -> None:
        if num_inputs < 1:
            raise ValueError("num_inputs must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if neurons_per_class < 1:
            raise ValueError("neurons_per_class must be positive")

        self.num_inputs = int(num_inputs)
        self.num_classes = int(num_classes)
        self.neurons_per_class = int(neurons_per_class)
        self.num_outputs = self.num_classes * self.neurons_per_class

        settings = {**GOLDEN_CONFIG, **(config or {})}
        self.learning_rate = float(settings["learning_rate"])
        self.phase_flexibility = float(settings["phase_flexibility"])
        self.lateral_strength = float(settings["lateral_strength"])
        self.input_threshold = float(settings["input_threshold"])
        self.kerr_constant = float(settings["kerr_constant"])
        self.system_energy = float(settings["system_energy"])

        self.init_lr = self.learning_rate
        self.init_flex = self.phase_flexibility
        self.time_steps = int(time_steps)
        self.encoding = PhasicEncoding(encoding)
        self.phase_init = phase_init
        self.balance_classes = bool(balance_classes)

        self.seed = seed
        self._rng = np.random.default_rng(seed)

        magnitude = np.full((self.num_inputs, self.num_outputs), 0.05)
        if phase_init == "random":
            phases = self._rng.uniform(0.0, 2.0 * np.pi, size=magnitude.shape)
        elif phase_init == "zero":
            phases = np.zeros_like(magnitude)
        else:
            raise ValueError(f"unknown phase_init {phase_init!r}")
        self.W_in = magnitude * np.exp(1j * phases)

        # Lateral coupling starts as self-coupling only and is zeroed on the
        # diagonal after every update, so what survives training is the
        # off-diagonal binding between neurons of the same class.
        self.W_lat = np.eye(self.num_outputs, dtype=complex) * self.lateral_strength

        self._class_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.samples_seen = 0

    # -- learning schedule -------------------------------------------------

    def decay_learning_rate(self, progress: float) -> None:
        """Anneal plasticity linearly over training.

        High plasticity early lets a class be learned from one presentation;
        decaying it later stops the last samples overwriting everything before
        them. ``progress`` runs 0 to 1.
        """
        factor = 1.0 - (float(progress) * 0.9)
        self.learning_rate = self.init_lr * factor
        self.phase_flexibility = self.init_flex * factor

    # -- forward -----------------------------------------------------------

    def get_phasic_input(self, feature_vector: np.ndarray) -> np.ndarray:
        """Encode a feature vector as a complex input wave."""
        return to_phasic(
            feature_vector,
            encoding=self.encoding,
            threshold=self.input_threshold,
        )

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Unitary L2 clamp -- the astrocyte energy budget.

        Total energy is held at or below ``system_energy``, so neurons compete
        for a fixed pool rather than saturating independently. Clamping only on
        the way down preserves contrast instead of clipping it.
        """
        energy = np.linalg.norm(state)
        if energy > 0:
            scale = self.system_energy / energy
            if scale < 1.0:
                state = state * scale
        return state

    def resonate(self, input_wave: np.ndarray) -> np.ndarray:
        """Run the resonant loop and return the settled cortical state."""
        state = np.zeros(self.num_outputs, dtype=complex)
        for _ in range(self.time_steps):
            feedforward = np.dot(input_wave, self.W_in)
            feedback = np.dot(state, self.W_lat)
            state = feedforward + feedback

            # Kerr non-linearity: intensity shifts the medium's refractive
            # index, so a strong signal twists its own phase and self-focuses.
            magnitudes = np.abs(state)
            phases = np.angle(state)
            state = magnitudes * np.exp(1j * (phases + self.kerr_constant * magnitudes**2))

            state = self.normalize_state(state)
        return state

    def predict(self, feature_vector: np.ndarray) -> Prediction:
        """Score one sample without learning from it."""
        state = self.resonate(self.get_phasic_input(feature_vector))
        energies = np.abs(state) ** 2
        pooled = energies.reshape(self.num_classes, self.neurons_per_class).sum(axis=1)
        return Prediction(
            label=int(np.argmax(pooled)),
            energies=pooled,
            total_energy=float(np.sum(energies)),
        )

    # -- learning ----------------------------------------------------------

    def observe(
        self, feature_vector: np.ndarray, label: int, *, weight: float = 1.0
    ) -> Prediction:
        """Score one sample and learn from it. The online path.

        One presentation, one update, no replay. Returns what the cortex
        believed *before* the update, which is the only reading usable as an
        online accuracy estimate -- scoring after the update would be scoring
        the answer against a model that had already been told it.

        ``weight`` scales this sample's plasticity. The default of 1.0 leaves
        the learning rule exactly as validated. Below 1.0 the sample moves the
        weights less, which is how a partially-observed outcome earns
        proportionally less influence -- a funded deal watched for 20 days of a
        90-day term is weaker evidence than one watched to maturity, and
        letting it push as hard is how a model of a growing book comes out
        optimistic.

        Exclusive classes only. When a sample can carry several independent
        labels at once -- one merchant file drawing offers from three lenders
        and declines from two -- use :meth:`observe_multi`.
        """
        if not 0 <= label < self.num_classes:
            raise ValueError(f"label {label} outside [0, {self.num_classes})")

        input_wave, prediction = self._forward(feature_vector)

        # Single-label damping targets whichever class won wrongly, which is
        # the rule the published result was measured with.
        negatives = () if prediction.label == label else (prediction.label,)
        self._learn(input_wave, (label,), negatives, weight=weight)

        self._class_counts[label] += 1
        self.samples_seen += 1
        return prediction

    def observe_multi(
        self,
        feature_vector: np.ndarray,
        positives: Iterable[int],
        negatives: Iterable[int] = (),
        *,
        weight: float = 1.0,
    ) -> Prediction:
        """Learn from a sample carrying several independent outcomes at once.

        The shape lender acceptance actually has: one merchant file is shopped
        to a handful of lenders and each answers for itself. Three offers and
        two declines is five observations about one file, not one label.

        **A class in neither set is unobserved, and is left alone.** That
        distinction is the whole reason this method exists rather than a
        positives-only signature. A lender the file was never submitted to has
        not declined it; training it as a negative teaches the cortex that
        every lender you did not approach says no, which is both false and
        exactly the direction that makes a ranking useless -- it would learn
        your submission habits rather than the lenders' credit boxes.

        Sharing one input representation across every lender while keeping a
        per-lender output block mirrors how the existing reduced-form model is
        built, where the slope on expected loss is pooled across lenders and
        only the intercept is fitted per lender. Thin per-lender data is the
        reason there, and it is the reason here.

        Args:
            positives: Classes that fired for this sample -- lenders that made
                an offer.
            negatives: Classes observed *not* to fire -- lenders that declined.
            weight: Plasticity scale for the whole sample, as in :meth:`observe`.
        """
        positive = tuple(dict.fromkeys(int(c) for c in positives))
        negative = tuple(dict.fromkeys(int(c) for c in negatives))

        for cls in (*positive, *negative):
            if not 0 <= cls < self.num_classes:
                raise ValueError(f"class {cls} outside [0, {self.num_classes})")
        overlap = set(positive) & set(negative)
        if overlap:
            raise ValueError(
                f"classes {sorted(overlap)} are both positive and negative for "
                "one sample; an outcome cannot be an accept and a decline"
            )
        if not positive and not negative:
            raise ValueError("observe_multi needs at least one observed outcome")

        input_wave, prediction = self._forward(feature_vector)
        self._learn(input_wave, positive, negative, weight=weight)

        for cls in positive:
            self._class_counts[cls] += 1
        self.samples_seen += 1
        return prediction

    def _forward(self, feature_vector: np.ndarray) -> tuple[np.ndarray, Prediction]:
        """One resonant pass. Returns the input wave and the reading."""
        input_wave = self.get_phasic_input(feature_vector)
        state = self.resonate(input_wave)
        energies = np.abs(state) ** 2
        pooled = energies.reshape(self.num_classes, self.neurons_per_class).sum(axis=1)
        return input_wave, Prediction(
            label=int(np.argmax(pooled)),
            energies=pooled,
            total_energy=float(np.sum(energies)),
        )

    def _gain_for(self, label: int, weight: float = 1.0) -> float:
        """How hard this sample should push, for one class.

        Combines the sample weight with class balancing into a single
        multiplier applied to *both* halves of the learning rule.

        The sample weight is clipped to [0, 4]: a caller computing
        ``observed_days / term_days`` on bad data should lose that sample's
        influence, not the whole model. Class balancing is clipped for the
        related reason that an unseen class would otherwise divide by a count
        of zero and consume the whole column.
        """
        scale = float(np.clip(weight, 0.0, 4.0))
        if not self.balance_classes or self.samples_seen == 0:
            return scale
        seen = self._class_counts[label]
        if seen == 0:
            return 4.0 * scale
        mean = float(np.mean(self._class_counts[self._class_counts > 0]))
        return float(np.clip(mean / seen, 0.25, 4.0)) * scale

    def _plasticity_for(self, label: int, weight: float = 1.0) -> float:
        """This sample's effective learning rate -- the magnitude half.

        Without balancing this is the current rate times the sample weight.
        With it, a class the cortex has seen rarely gets proportionally more
        plasticity, so a book that is 88% non-defaults does not simply learn to
        say "non-default".
        """
        return self.learning_rate * self._gain_for(label, weight)

    def _flexibility_for(self, label: int, weight: float = 1.0) -> float:
        """This sample's effective phase flexibility -- the phase half.

        Scaled by the same gain as the learning rate, and that pairing is
        load-bearing rather than tidy. This is a *phase*-Hebbian rule: the
        rotation toward zero phase is what brings a class's neurons into step
        with an input, and it is arguably the more important half of learning
        here. Scaling only the magnitude would leave a sample down-weighted for
        partial observation still rotating the phases at full strength -- so a
        deal watched 20 days of 90 would move the model almost as much as one
        watched to maturity, which is the exact bias the weight exists to
        prevent.
        """
        return self.phase_flexibility * self._gain_for(label, weight)

    def _columns_for(self, cls: int) -> np.ndarray:
        """The output-neuron columns belonging to one class."""
        start = cls * self.neurons_per_class
        return np.arange(start, start + self.neurons_per_class)

    def _learn(
        self,
        input_wave: np.ndarray,
        positives: Sequence[int],
        negatives: Sequence[int],
        *,
        weight: float = 1.0,
    ) -> None:
        """Apply the local Hebbian update for one presentation.

        Classes in neither sequence are untouched -- that is what makes an
        unobserved outcome different from a negative one.

        Order matters: every positive is reinforced before any negative is
        damped, so a class appearing as a positive here cannot have its own
        growth eroded within the same update by a different class's damping
        pass over the shared active inputs.
        """
        active = np.flatnonzero(np.abs(input_wave) > 0.1)

        if active.size:
            for cls in positives:
                rate = self._plasticity_for(cls, weight)
                flex = self._flexibility_for(cls, weight)

                # 1. Feedforward phase-Hebbian: rotate the active weights
                #    toward zero phase and grow them, so this class's neurons
                #    come into step with this input and resonate harder next
                #    time.
                columns = self._columns_for(cls)
                block = self.W_in[np.ix_(active, columns)]
                block = block * np.exp(-1j * flex * np.angle(block))
                self.W_in[np.ix_(active, columns)] = block * (1.0 + rate)

                # 2. Lateral Hebbian: bind this class's neurons to each other
                #    so they fire as one assembly.
                start = cls * self.neurons_per_class
                stop = start + self.neurons_per_class
                self.W_lat[start:stop, start:stop] *= 1.0 + rate

            # 3. Damping: scatter the phases of each observed negative and
            #    shrink its weights. Random scatter rather than a directed
            #    correction is what keeps this a local rule -- there is no
            #    gradient telling it which way to move.
            for cls in negatives:
                rate = self._plasticity_for(cls, weight)
                flex = self._flexibility_for(cls, weight)
                columns = self._columns_for(cls)
                block = self.W_in[np.ix_(active, columns)]
                noise = self._rng.uniform(-1.0, 1.0, size=block.shape)
                block = block * np.exp(1j * flex * noise)
                self.W_in[np.ix_(active, columns)] = block * (1.0 - rate)

        # Bound the weights so repeated growth cannot run away, preserving
        # phase and clipping only magnitude.
        self.W_in = np.clip(np.abs(self.W_in), 0.0, 1.0) * np.exp(1j * np.angle(self.W_in))
        self.W_lat = np.clip(np.abs(self.W_lat), 0.0, 0.5) * np.exp(1j * np.angle(self.W_lat))
        np.fill_diagonal(self.W_lat, 0.0)

    # -- compatibility -----------------------------------------------------

    def process(
        self, feature_vector: np.ndarray, label: int, train: bool = True
    ) -> tuple[bool, int, float]:  # noqa: FBT001, FBT002
        """Original call shape: ``(was_correct, predicted_label, total_energy)``.

        Retained so the MNIST benchmark keeps working unchanged as the
        regression guard. New code wants :meth:`observe` or :meth:`predict`,
        which return the margin and the full energy distribution.
        """
        prediction = self.observe(feature_vector, label) if train else self.predict(feature_vector)
        return prediction.label == label, prediction.label, prediction.total_energy

    #: Historical name. This was never image-specific.
    process_image = process

    # -- persistence -------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Write the trained cortex to a ``.npz``.

        Complex matrices survive ``.npz`` natively, so weights round-trip
        exactly rather than through a lossy text encoding. Everything needed to
        reconstruct and to audit travels with them: the physics, the encoding,
        the seed, and how many samples of each class were seen.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            format_version=_FORMAT_VERSION,
            W_in=self.W_in,
            W_lat=self.W_lat,
            class_counts=self._class_counts,
            num_inputs=self.num_inputs,
            num_classes=self.num_classes,
            neurons_per_class=self.neurons_per_class,
            time_steps=self.time_steps,
            samples_seen=self.samples_seen,
            seed=-1 if self.seed is None else self.seed,
            encoding=str(self.encoding),
            phase_init=self.phase_init,
            balance_classes=self.balance_classes,
            learning_rate=self.learning_rate,
            phase_flexibility=self.phase_flexibility,
            lateral_strength=self.lateral_strength,
            input_threshold=self.input_threshold,
            kerr_constant=self.kerr_constant,
            system_energy=self.system_energy,
            init_lr=self.init_lr,
            init_flex=self.init_flex,
        )
        return target if target.suffix else target.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path | str) -> QuantumCortex:
        """Reconstruct a cortex saved by :meth:`save`."""
        source = Path(path)
        if not source.exists() and not source.suffix:
            source = source.with_suffix(".npz")
        with np.load(source, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version > _FORMAT_VERSION:
                raise ValueError(
                    f"{source} was written by format version {version}; "
                    f"this build understands up to {_FORMAT_VERSION}"
                )
            seed = int(data["seed"])
            cortex = cls(
                num_inputs=int(data["num_inputs"]),
                num_classes=int(data["num_classes"]),
                neurons_per_class=int(data["neurons_per_class"]),
                config={
                    "learning_rate": float(data["learning_rate"]),
                    "phase_flexibility": float(data["phase_flexibility"]),
                    "lateral_strength": float(data["lateral_strength"]),
                    "input_threshold": float(data["input_threshold"]),
                    "kerr_constant": float(data["kerr_constant"]),
                    "system_energy": float(data["system_energy"]),
                },
                seed=None if seed < 0 else seed,
                encoding=PhasicEncoding(str(data["encoding"])),
                phase_init=str(data["phase_init"]),  # type: ignore[arg-type]
                balance_classes=bool(data["balance_classes"]),
                time_steps=int(data["time_steps"]),
            )
            cortex.W_in = np.array(data["W_in"])
            cortex.W_lat = np.array(data["W_lat"])
            cortex._class_counts = np.array(data["class_counts"])
            cortex.samples_seen = int(data["samples_seen"])
            cortex.init_lr = float(data["init_lr"])
            cortex.init_flex = float(data["init_flex"])
        return cortex

    # -- introspection -----------------------------------------------------

    def class_counts(self) -> dict[int, int]:
        """How many samples of each class this cortex has learned from."""
        return {c: int(n) for c, n in enumerate(self._class_counts)}

    def describe(self) -> dict[str, Any]:
        """A flat summary, for a model registry payload or an audit line."""
        return {
            "num_inputs": self.num_inputs,
            "num_classes": self.num_classes,
            "neurons_per_class": self.neurons_per_class,
            "time_steps": self.time_steps,
            "encoding": str(self.encoding),
            "phase_init": self.phase_init,
            "balance_classes": self.balance_classes,
            "seed": self.seed,
            "samples_seen": self.samples_seen,
            "class_counts": self.class_counts(),
            "physics": {
                "learning_rate": self.init_lr,
                "phase_flexibility": self.init_flex,
                "lateral_strength": self.lateral_strength,
                "input_threshold": self.input_threshold,
                "kerr_constant": self.kerr_constant,
                "system_energy": self.system_energy,
            },
        }
